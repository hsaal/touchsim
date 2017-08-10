import numpy as np
import re
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.morphology import binary_fill_holes,binary_dilation
from skimage.measure import label, find_contours
from skimage.morphology import thin
from matplotlib import path
from scipy.io import loadmat

from .constants import hand_tags,hand_orig,hand_pxl_per_mm,hand_theta

class Surface(object):

    def __init__(self,**args):
        self.orig = args.get('orig',np.array([0., 0.]))
        self.pxl_per_mm = args.get('pxl_per_mm',1.)
        self.theta = args.get('theta',0.)
        self.rot2hand = np.array([[np.cos(self.theta), -np.sin(self.theta)],
            [np.sin(self.theta), np.cos(self.theta)]])
        self.rot2pixel = np.array([[np.cos(-self.theta), -np.sin(-self.theta)],
            [np.sin(-self.theta), np.cos(-self.theta)]])
        self.tags = args.get('tags',None)

        self.outline = args.get('outline',None)
        if self.outline is None:
            self.label = None
        else:
            if args.get('preproc',True):
                self.outline = np.int64(thin(self.outline))
                self.label,self.num_reg = label(self.outline,connectivity=1,background=1,return_num=True)

                self.boundary = []
                for i in range(self.num_reg-1):
                    dd = distance_transform_edt(np.flipud(self.label==(i+2)))
                    xy = find_contours(dd,1)
                    self.boundary.append(xy[0][:,::-1])

        self.construct_dist_matrix()

    def add_tags(self,tags):
        self.tags = tags

    def hand2pixel(self,locs):
        return np.dot(locs,self.rot2pixel)*self.pxl_per_mm + self.orig

    def pixel2hand(self,locs):
        return np.dot((locs-self.orig)/self.pxl_per_mm,self.rot2hand)

    def tag2idx(self,tag):
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

    def sample_uniform(self,idx,dens):
        if self.outline is None:
            raise RuntimeError("Cannot sample from surface without border.")

        b = bbox(self.boundary[idx])
        xy = np.mgrid[b[0]:b[2]+1./dens:1./dens,b[1]:b[3]+1./dens:1./dens]
        xy = xy.reshape(2,xy.shape[1]*xy.shape[2]).T
        xy += np.random.randn(xy.shape[0],xy.shape[1])/dens/5.
        p = path.Path(self.boundary[idx])
        ind = p.contains_points(xy);
        xy = xy[ind,:]

        return self.pixel2hand(xy)

    def construct_dist_matrix(self):
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

    def distance(self,xy_pin,xy_aff):
        if self.D is None:
            dx = xy_pin[:,0:1] - xy_aff[:,0:1].T
            dy = xy_pin[:,1:2] - xy_aff[:,1:2].T
            return np.sqrt(dx**2 + dy**2)
        else:
            lin_idx = np.arange(self.outline.size).reshape(self.outline.T.shape)

            xyp = np.rint(self.hand2pixel(xy_pin)).astype(np.int64)
            xyp = lin_idx[xyp[:,0],xyp[:,1]]
            xya = np.rint(self.hand2pixel(xy_aff)).astype(np.int64)
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
    return np.hstack((np.min(xy,axis=0),np.max(xy,axis=0)))

null_surface = Surface()
hand_surface = Surface(outline=(1-loadmat('../base/GUI/hand')['hand']),
        orig=hand_orig,pxl_per_mm=hand_pxl_per_mm,
        theta=hand_theta)
hand_surface.add_tags(hand_tags)
