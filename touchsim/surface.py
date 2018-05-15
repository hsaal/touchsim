import numpy as np
import re
import os.path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.morphology import binary_fill_holes,binary_dilation
from skimage.measure import label, find_contours
from skimage.morphology import thin
from matplotlib import path
from PIL import Image

from .constants import hand_tags,hand_orig,hand_pxl_per_mm,hand_theta,hand_density

class Surface(object):
    """A class representing a finite surface, which can be subdivided into
    separate regions, and on which Afferent objects can be placed.
    """

    def __init__(self,**args):
        """Initializes a Surface object.

        Kwargs:
            orig (array): Origin of coordinate system in pixel space (default: [0,0]).
            pxl_per_mm (float): Conversion factor from pixels to millimeters (default: 1.).
            theta (float): Angle of first axis in pixel space in radians (default: 0.).
            filename (string): Filename (with optional path) to an image file
                containing surface outline (for example: 'surfaces/hand.tiff',
                default: None).
            outline (array): 2D matrix containing outline of surface (and subregions);
                if set to None (the default), and no image filename is given,
                surface will represent infinite sheet. This is intended to be used as an
                alternative to specifying a filename, if an outline is already present
                in array form.
            tags (list): List of tuples with 3 strings each, denoting 1) coarse
                region, 2) sub-region, and 3) density tag (default: all empty strings).
            density (dict): Mapping between tuples containing 1) string denoting
                afferent class and 2) string denoting density tag, and float
                denoting afferent density in cm^2 (default: 10. for each mapping).
        """
        self.orig = args.get('orig',np.array([0., 0.]))
        self.pxl_per_mm = args.get('pxl_per_mm',1.)
        self.theta = args.get('theta',0.)
        self.rot2hand = np.array([[np.cos(self.theta), -np.sin(self.theta)],
            [np.sin(self.theta), np.cos(self.theta)]])
        self.rot2pixel = np.array([[np.cos(-self.theta), -np.sin(-self.theta)],
            [np.sin(-self.theta), np.cos(-self.theta)]])

        im = args.get('filename',None)
        if im is None:
            self.outline = args.get('outline',None)
        else:
            self.outline = image2outline(im,args.get('thres',250))

        if self.outline is None:
            self.label = None
            self.num = 0
        else:
            self.outline = np.int64(thin(self.outline))
            self.label,self.num = label(self.outline,connectivity=1,background=1,\
                return_num=True)
            self.num -= 1
            self.boundary = []
            self.centers = []
            for i in range(self.num):
                dd = distance_transform_edt(np.flipud(self.label==(i+2)))
                xy = find_contours(dd,1)
                self.boundary.append(xy[0][:,::-1])
                self.centers.append(np.mean(xy[0][:,::-1],axis=0))

        self.construct_dist_matrix()
        self.tags = args.get('tags',[('','','') for i in range(self.num)])
        self.density = args.get('density',{('SA1',''):10.,('RA',''):10., ('PC',''):10.})

    def hand2pixel(self,locs):
        """Transforms from surface coordinates to pixel coordinates.

        Args:
            locs (array): 2D array of coordinates in surface space.

        Returns:
            2D array with transformed coordinates in pixel space.
        """
        return np.dot(locs,self.rot2pixel)*self.pxl_per_mm + self.orig

    def pixel2hand(self,locs):
        """Transforms from pixel coordinates to surface coordinates.

        Args:
            locs (array): 2D array of coordinates in pixel space.

        Returns:
            2D array with transformed coordinates in surface space.
        """
        return np.dot((locs-self.orig)/self.pxl_per_mm,self.rot2hand)

    def tag2idx(self,tag):
        """Maps from surface tags to region ID numbers.

        Args:
            tag (str): Region tag.

        Returns:
            List of region ID numbers matching specified tag.
        """
        if self.tags is None:
            raise RuntimeError("No tags set for this surface.")
        if tag is None:
            return range(len(self.boundary))
        else:
            match = re.findall('[dDpPwWmMdDfFtT]\d?',tag)
            idx = [i for i,x in enumerate(self.tags) if x[0]==match[0]]
            if len(match)>1:
                return set(idx).intersection(
                    [i for i,x in enumerate(self.tags) if x[1]==match[1]])
            return idx

    def sample_uniform(self,id_or_tag,**args):
        """Samples locations from within specified region.

        Args:
            id_or_tag (int or str): region ID number or tag identifying a unique
                region.

        Kwargs:
            num (int): Number of locations to sample (default: None).
            density (float): Density of locations to be sampled expressed as
                locations per cm^2 (default: SA1 density for specified region).
                This parameter will only be evaluated if num is not given / set to None.

        Returns:
            2D array of coordinates in surface space.
        """
        if self.outline is None:
            raise RuntimeError("Cannot sample from surface without border.")

        if type(id_or_tag) is str:
            idx = self.tag2idx(tag)[0]
        else:
            idx = id_or_tag

        num = args.get('num',None)
        if num is None:
            dens = args.get('density',self.density[('SA1',self.tags[idx][2])])
            dist = np.sqrt(dens)/10./self.pxl_per_mm
            b = bbox(self.boundary[idx])
            xy = np.mgrid[b[0]:b[2]+1./dist:1./dist,b[1]:b[3]+1./dist:1./dist]
            xy = xy.reshape(2,xy.shape[1]*xy.shape[2]).T
            xy += np.random.randn(xy.shape[0],xy.shape[1])/dist/5.
            p = path.Path(self.boundary[idx])
            ind = p.contains_points(xy);
            xy = xy[ind,:]

        else:
            xy = np.zeros((num,2))
            for i in range(num):
                xy[i] = rejection_sample(self.boundary[idx])

        return self.pixel2hand(xy)

    def construct_dist_matrix(self):
        """Constructs matrix of pair-wise distances between all pixels contained
        in the surface. This method is executed automatically when the outline
        variable is set during construction of the Surface object.
        """
        if self.outline is None:
            self.D = None
            return

        hand = np.fliplr(self.outline.T)
        hand = binary_fill_holes(hand)
        hand = binary_dilation(hand)

        idx = np.flatnonzero(hand.flatten())
        pos = np.reshape(np.arange(hand.size),hand.shape)

        shifts = [(1,0),(-1,0),(0,1),(0,-1),(1, 1),(-1,-1),(1,-1),(-1, 1)]
        dist = [1., 1., 1., 1., np.sqrt(2.), np.sqrt(2.), np.sqrt(2.), np.sqrt(2.)]

        nodes = np.array([],dtype=np.int64).reshape(0,2)
        weights = np.array([])
        for i in range(len(dist)):
            mask = np.logical_and(hand,np.roll(hand,shifts[i],axis=(0,1)))
            ind = np.flatnonzero(mask.flatten()[idx])
            pos_shift = np.roll(pos,shifts[i],axis=(0,1)).flatten()
            nodes = np.vstack((nodes,
                np.hstack((idx[ind][:,None],pos_shift[idx[ind]][:,None]))))
            weights = np.concatenate((weights, np.tile(dist[i]/self.pxl_per_mm,ind.size)))

        self.D = csr_matrix((weights,(nodes[:,0],nodes[:,1])),shape=(hand.size,hand.size))


    def distance(self,xy1,xy2):
        """Computes the shortest distance between pairwise locations on the surface.

        Args:
            xy1 (2D array): Origin location(s) in surface space.
            xy2 (2D array): Destination location(s) in surface space.

        Returns:
            2D array containing all pairwise distances between origin and
            destination locations.
        """
        if self.D is None:
            dx = xy1[:,0:1] - xy2[:,0:1].T
            dy = xy1[:,1:2] - xy2[:,1:2].T
            return np.sqrt(dx**2 + dy**2)
        else:
            lin_idx = np.arange(self.outline.size).reshape(self.outline.T.shape)

            xyp = np.rint(self.hand2pixel(xy1)).astype(np.int64)
            xyp = lin_idx[xyp[:,0],xyp[:,1]]
            xya = np.rint(self.hand2pixel(xy2)).astype(np.int64)
            xya = lin_idx[xya[:,0],xya[:,1]]

            flip = False
            if xyp.size>xya.size:
                flip = True
                xyp,xya = xya,xyp

            D = dijkstra(self.D,directed=False,indices=xyp)
            D = D[:,xya]

            if flip:
                D = D.T

            return D


def bbox(xy):
    """Calculates bounding box for arbitrary boundary.
    """
    return np.hstack((np.min(xy,axis=0),np.max(xy,axis=0)))

def rejection_sample(boundary):
    """Samples a single location from within arbitrary boundary.
    """
    b = bbox(boundary)
    p = path.Path(boundary)
    inside = False
    while not inside:
        xy = np.atleast_2d(b[[0,2]]) + np.random.random((1,2))*\
            (np.atleast_2d(b[[1,3]])-np.atleast_2d(b[[0,2]]))
        inside = p.contains_point(xy.T)
    return xy

def image2outline(filename,thres=250):
    """Converts image to greyscale and thresholds to generate binary outline.
    """
    im = Image.open(filename)
    im = im.convert('L',dither=None)
    outline = np.array(im)<thres
    return outline

null_surface = Surface()
hand_surface = Surface(filename=(os.path.dirname(os.path.dirname(__file__))\
        + '/surfaces/hand.tiff'),orig=hand_orig,pxl_per_mm=hand_pxl_per_mm,
        theta=hand_theta,density=hand_density,tags=hand_tags)
