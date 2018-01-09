'''file to generate the responses for a realistic density hand population 
with uniform placement'''


import touchsim as ts
import numpy as np
import scipy.io as io
import pickle as pk
import random
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt

def inputs_rdu(number):
    
    # create random seed
    random.seed(number)
    
    #load the saved hand_pop object
    with open('rd_hand_pop', 'rb') as f:
        hand_pop = pk.load(f)

    # check hand_pop location
    a = hand_pop.location
    
    # create empty array for responses
    x = np.empty((len(a),0))
    
    # create empty array for stimuli locations
    stim = np.empty((0,2))
    
    # create empty array for radius
    stim_radius = np.empty((0,0))
    
    # get the region size data for the weights
    p = region_sizes(hand_pop)
    
    # labels is the variables
    labels = np.arange(20)
    
    hand_surface = hand_pop.surface
    
    # get the bounding boxes
    boundaries = hand_pop.surface.boundary
    
    # for loop to create stimuli and responses
    for i in range(10):
        
        inarea = 0
        
        # selects one of the areas for the point to be located. P is the density, region sizes
        area_num = np.random.choice(labels, 1, p.tolist)
        
        # get the integer from array
        area_number = area_num[0]
        
        # get the boundary coordinates
        area_boundary = boundaries[area_number]
        #xv = area_boundary[:,0]
        #yv = area_boundary[:,1]
        
        # get bounding box
        boundary_box = bounding_box(area_boundary)
        
        while inarea == 0:
            
            coord = random_coordinate(boundary_box)
            
            coord = np.reshape(coord, (-1, 2))
            
            coord_xy = coord
            
            # change coordinate system to plot
            coord_xy = pixel2hand(hand_surface,coord)
        
            # check coordinate is in the bounding box 
            path = mpltPath.Path(area_boundary.tolist()) 
            inside = path.contains_points(coord) 
            
            ''' visualise point check'''
            # 
            #plt.scatter(area_boundary[:,0],area_boundary[:,1])
           # plt.scatter(coord[:,0],coord[:,1])
            #plt.show()
            
            if inside == True:
                inarea = 1;
                
        # create radius value- will be stored
        radius= 5+(10*random.random())
        
        # create stimulus at sample afferent
        s = ts.stim_ramp(amp=0.5,len=0.2,loc=coord_xy.tolist(),fs=1000,pin_radius=radius)
        
        # add stimuli to stimuli matrix
        stim = np.append(stim, coord, axis=0)
        
        # add radius to matrix
        stim_radius = np.append(stim_radius, radius)
        
        # calculate response to stimulus located at sampled afferent
        r = hand_pop.response(s).rate()

        # add response to input response matrix
        x = np.append(x, r, axis=1)
        
    
    # turn into saveable data type
    stim_data = {'stim' : stim.tolist()}
    
    # turn into saveable data type
    data = {'x' : x.tolist()}
    
    # turn into saveable data type
    radius_data = {'stim_radius' : stim_radius.tolist()}
    
    # save inputs
    io.savemat('inputs_rdu',data)
    
    # save inputs
    io.savemat('stimuli_rdu',stim_data)

    # save inputs
    io.savemat('stim_radius_rdu',radius_data)
    
    
''' extra methods for input generation'''
    
''' calculates the sizes of each region in the shape'''
def region_sizes(hand_pop):
    
    # get regionprop propeties
    d = hand_pop.surface.boundary
    
    #  pre-allocate response matrix for the area sizes
    a = np.zeros((20,1))
    
    # calculate the area of each. Needs to be in the form:
    def polyarea(x,y):
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    
    # for each area find the size, assign to vector A
    for i in range(len(a)):
        
        c = d[i]
           
        a[i] = polyarea(c[:,0],c[:,1])

    # calculate size of areas as a percentage
    percentage = np.multiply(a,(100/np.sum(a)))
    
    return percentage


 # creates a random coordinate within the bounding box for that finger area.
def random_coordinate(bb_coord):

    # get maximum and minimum coordinates of the bounding box
    x_min = bb_coord[0,0]
    x_max = bb_coord[0,1]
    y_min = bb_coord[1,0]
    y_max = bb_coord[1,1]

    # calculate random coordinate within the bounding box
    x = np.multiply((x_max-x_min),random.random()) + x_min
    y = np.multiply((y_max-y_min),random.random()) + y_min
    
# assign random coordinate to value
    c = [x,y];
    
    return c

 # creates a random coordinate within the bounding box for that finger area.
def random_coordinate2(list_num, hand_pop):

   # use sample_uniform method
   f = ts.Surface.sample_uniform(hand_pop.surface,list_num,0.1)
   
   # assign random coordinate to value
   c = f[0,:]
   
   return c

''' create bounding box of set of point'''
def bounding_box(points):
    """
    [xmin xmax]
    [ymin ymax]
    """
    a = np.zeros([2,2])
    a[:,0] = np.min(points, axis=0)
    a[:,1] = np.max(points, axis=0)
    return a

#''' visualisation code'''
## Visualise the whole hand
#
#with open('rd_hand_pop', 'rb') as f:
#        hand_pop = pk.load(f)
#        
#boundaries = hand_pop.surface.boundary
#
#for i in range(20):
#    area_boundary = boundaries[i]
#    plt.scatter(area_boundary[:,0],area_boundary[:,1])
#    
##plt.scatter(h[:,0],h[:,1])

def pixel2hand(self,locs):
        """ Transforms from pixel coordinates to hand coordinates.
        """
        return np.dot((locs-self.orig)/self.pxl_per_mm,self.rot2hand)
    