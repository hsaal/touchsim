import numpy as np
import re
import os.path
import warnings
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.morphology import binary_fill_holes,binary_dilation
from skimage.measure import label, find_contours, regionprops
from skimage.morphology import thin
from matplotlib import path
from PIL import Image

from .constants import hand_tags,hand_orig,hand_pxl_per_mm,hand_theta,hand_density

default_density = {('SA1',''):10., ('RA',''):10., ('PC',''):10.}

class Surface(object):
    """A representation of a finite surface, which can be subdivided into
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
        self.meta = args
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
            self.num = 0
            self.density = {}
            self.tags = []
        else:
            self.outline = np.int64(thin(self.outline))
            labels,self.num = label(self.outline,connectivity=1,background=1,\
                return_num=True)
            regions = regionprops(np.flipud(labels))
            self.num -= 1
            self.boundary = []
            self._centers = []
            self._coords = []
            self._area = []
            for i in range(self.num):
                dd = distance_transform_edt(np.flipud(labels==(i+2)))
                xy = find_contours(dd,1)
                if len(xy)==0:
                    continue
                self.boundary.append(xy[0][:,::-1])
                self._centers.append(np.mean(xy[0][:,::-1],axis=0))
                self._area.append(regions[i+1].area)
                self._coords.append(regions[i+1].coords[:,::-1])
            self.num = len(self.boundary)
            self._centers = np.array(self._centers)
            self._area = np.array(self._area)

            self.bbox_min = np.zeros((self.num,2))
            self.bbox_max = np.zeros((self.num,2))
            for i in range(self.num):
                bb = bbox(self.boundary[i])
                self.bbox_min[i] = bb[0]
                self.bbox_max[i] = bb[1]

            self.tags = args.get('tags',['' for i in range(self.num)])
            self.density = args.get('density',default_density)

        self.construct_dist_matrix()


    @property
    def density(self):
        return self._density

    @density.setter
    def density(self,density):
        self._density = {}
        wflag = False
        for k in density.keys():
            idx = self.tag2idx(k[1])
            if len(idx)==0:
                warnings.warn("No matches found for tag '" + k[1] + "'.",
                    stacklevel=2)
            for i in idx:
                if (k[0],i) in self._density.keys():
                    wflag = True
                self._density[(k[0],i)] = density[k]
        if wflag:
            warnings.warn("Density assignment not unique. Please check tags.",
                stacklevel=2)
        if len(self._density.keys())<self.num*3:
            warnings.warn("Some regions have no density assigned.",
                stacklevel=2)

    def __str__(self):
        return 'Surface with ' + str(self.num) + ' regions.'

    def __len__(self):
        return len(self.num)

    @property
    def centers(self):
        return self.pixel2hand(self._centers)

    @property
    def area(self):
        return self._area/self.pxl_per_mm**2.

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
            return list(range(self.num))
        else:
            idx = [i for i in range(self.num) if len(re.findall(tag,self.tags[i]))>0]
            return idx

    def locate(self,locs):
        """Maps from coordinates on a surface to region tags and ids.

        Args:
            locs (array): 2D array of coordinates in pixel space.

        Returns:
            Tuple containing a list of region tags and a vector of ids.
        """
        locs = np.atleast_2d(self.hand2pixel(locs))
        regions = -np.ones((locs.shape[0],),dtype=np.int8)

        for b in range(self.num):
            p = path.Path(self.boundary[b])
            ind = p.contains_points(locs)
            regions[ind] = b

        tags = []
        for l in range(locs.shape[0]):
            if regions[l]<0:
                tags.append('')
            else:
                tags.append(self.tags[regions[l]])

        return tags, regions

    def sample_uniform(self,id_or_tag,**args):
        """Samples locations from within specified region.

        Args:
            id_or_tag (int or str): region ID number or tag identifying a unique
                region.

        Kwargs:
            num (int): Number of locations to sample (default: None).
            density (float): Density of locations to be sampled expressed as
                locations per cm^2 (default: SA1 density for specified region(s)).
                This parameter will only be evaluated if num is not given / set to None.
            seed (int): Random number seed

        Returns:
            2D array of coordinates in surface space.
        """
        if self.outline is None:
            raise RuntimeError("Cannot sample from surface without border.")

        seed = args.get('seed',None)
        if seed is not None:
            np.random.seed(seed)

        if type(id_or_tag) is str or id_or_tag is None:
            idx = self.tag2idx(id_or_tag)
        elif type(id_or_tag) is int or type(id_or_tag) is np.int64:
            idx = [id_or_tag]
        else:
            idx = id_or_tag

        num = args.get('num',None)
        if num is None:
            xy_list = []
            for i in idx:
                dens = args.get('density',self.density[('SA1',i)])
                dist = np.sqrt(dens)/10./self.pxl_per_mm
                b = bbox(self.boundary[i])
                xy = np.mgrid[b[0,0]:b[1,0]+1./dist:1./dist,b[0,1]:b[1,1]+1./dist:1./dist]
                xy = xy.reshape(2,xy.shape[1]*xy.shape[2]).T
                xy += np.random.randn(xy.shape[0],xy.shape[1])/dist/5.
                p = path.Path(self.boundary[i])
                ind = p.contains_points(xy);
                xy = xy[ind,:]
                xy_list.append(xy)
            xy = np.concatenate(xy_list)

        else:
            xy = np.zeros((num,2))
            coords = np.concatenate([self._coords[i] for i in idx])
            for i in range(num):
                xy[i] = coords[np.random.randint(coords.shape[0])]

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
            # if infinite sheet, calculate length of straight line
            dx = xy1[:,0:1] - xy2[:,0:1].T
            dy = xy1[:,1:2] - xy2[:,1:2].T
            return np.sqrt(dx**2 + dy**2)
        else:
            # convert between 2D and 1D coordinate representation
            lin_idx = np.arange(self.outline.size).reshape(self.outline.T.shape)
            xyp = np.rint(self.hand2pixel(xy1)).astype(np.int64)
            xyp[xyp<0] = 0
            xyp[xyp[:,0]>=lin_idx.shape[0],0] = lin_idx.shape[0]-1
            xyp[xyp[:,1]>=lin_idx.shape[1],1] = lin_idx.shape[1]-1
            xyp = lin_idx[xyp[:,0],xyp[:,1]]

            xya = np.rint(self.hand2pixel(xy2)).astype(np.int64)
            xya[xya<0] = 0
            xya[xya[:,0]>=lin_idx.shape[0],0] = lin_idx.shape[0]-1
            xya[xya[:,1]>=lin_idx.shape[1],1] = lin_idx.shape[1]-1
            xya = lin_idx[xya[:,0],xya[:,1]]

            # flip for speeding up computations
            flip = False
            if xyp.size>xya.size:
                flip = True
                xyp,xya = xya,xyp

            D = dijkstra(self.D,directed=False,indices=xyp)
            D = D[:,xya]

            if np.any(np.isinf(D)):
                warnings.warn("At least one afferent or pin centre is outside " +
                    "surface boundary and will be ignored.",stacklevel=2)

            if flip:
                D = D.T

            return D

    def export(self,filename='surface.gen'):
        text_file = open(filename, "w")
        for i in range(self.num):
            text_file.write("%s %s\n" % (i,self.tags[i]))
            for p in self.boundary[i]:
                text_file.write("%s %s\n" % (p[0],p[1]))
            text_file.write("END\n")
        text_file.write("END\n")
        text_file.close()


def bbox(xy):
    """Calculates bounding box for arbitrary boundary.
    """
    return np.vstack((np.min(xy,axis=0),np.max(xy,axis=0)))

def image2outline(filename,thres=250):
    """Converts image to greyscale and thresholds to generate binary outline.
    """
    im = Image.open(filename)
    im = im.convert('L',dither=None)
    outline = np.array(im)<thres
    return outline

null_surface = Surface()
hand_surface = Surface(filename=(os.path.dirname(os.path.dirname(__file__))\
        + '/surfaces/hand.png'),orig=hand_orig,pxl_per_mm=hand_pxl_per_mm,
        theta=hand_theta,density=hand_density,tags=hand_tags)
