import h5py
import numpy as np
import sys
import cv2
import mpii

keys = ['imgname','center','scale','part','visible','istrain']
annot = {k:[] for k in keys}


train_fn = "train_joints.csv"
train_dl = np.array([l.strip() for l in open(train_fn).readlines()])
test_fn = "test_joints.csv"
test_dl = np.array([l.strip() for l in open(test_fn).readlines()])

for i, line in enumerate(train_dl):
    datum = line.split(',')

    joints = np.asarray([float(p) for p in datum[1:29]])
    joints = joints.reshape((int(len(joints) / 2), 2))
    jointsc = joints.copy()
    joints = np.append(joints,[[0,0],[0,0]],axis=0)
    joints = joints[[0,1,2,3,4,5,14,15,12,13,6,7,8,9,10,11],:]

    coords = joints

    img = cv2.imread('../images/'+datum[0]) 
    h, w, _ = img.shape
    s = float(datum[-3])

    jointsc = jointsc.astype(np.int32)
    #x, y, w, h = cv2.boundingRect(np.asarray([jointsc.tolist()]))
    #c = np.array([x+w/2, y+w/2])
    c = np.array([float(datum[-2]),float(datum[-1])])

    imgnameRef = 'LSP/images/'+datum[0]
    imgname = np.zeros(50)
    refname = str(imgnameRef)
    for i in range(len(refname)): imgname[i] = ord(refname[i])
    annot['imgname'] += [imgname]
    annot['center'] += [c]
    annot['scale'] += [s]                
    annot['part'] += [coords]
    annot['visible'] += [np.ones(16)]
    annot['istrain'] += [1]
        
for i, line in enumerate(test_dl):
    datum = line.split(',')

    joints = np.asarray([float(p) for p in datum[1:]])
    joints = joints.reshape((int(len(joints) / 2), 2))
    jointsc = joints.copy()
    joints = np.append(joints,[[0,0],[0,0]],axis=0)
    joints = joints[[0,1,2,3,4,5,14,15,12,13,6,7,8,9,10,11],:]
    
    coords = joints

    img = cv2.imread('../images/'+datum[0]) 
    h, w, _ = img.shape
    s = 0.8*max(w,h)/200

    jointsc = jointsc.astype(np.int32)
    #x, y, w, h = cv2.boundingRect(np.asarray([jointsc.tolist()]))
    #c = np.array([x+w/2, y+w/2])
    c = np.array([w/2,h/2])

    imgnameRef = 'LSP/images/'+datum[0]
    imgname = np.zeros(50)
    refname = str(imgnameRef)
    for i in range(len(refname)): imgname[i] = ord(refname[i])
    annot['imgname'] += [imgname]
    annot['center'] += [c]
    annot['scale'] += [s]                  
    annot['part'] += [coords]
    annot['visible'] += [np.ones(16)]
    annot['istrain'] += [0]


imgnameRef = mpii.annot['annolist'][0][0][0]['image'][:]
trainRef = mpii.annot['img_train'][0][0][0]

for idx in range(mpii.nimages):
    print "\r",idx,
    sys.stdout.flush()

    for person in range(mpii.numpeople(idx)):
        c,s = mpii.location(idx,person)
        if not c[0] == -1:
            # Add info to annotation list
            imgname = np.zeros(50)
            refname = str('mpii/images/'+imgnameRef[idx][0][0][0][0])
            for i in range(len(refname)): imgname[i] = ord(refname[i])
            annot['imgname'] += [imgname]
            annot['center'] += [c]
            annot['scale'] += [s]

            if mpii.istrain(idx) == True:
                # Part annotations and visibility
                coords = np.zeros((16,2))
                vis = np.zeros(16)
                for part in range(16):
                    coords[part],vis[part] = mpii.partinfo(idx,person,part)
                annot['part'] += [coords]
                annot['visible'] += [vis]
                annot['istrain'] += [1]
            else:
                annot['part'] += [-np.ones((16,2))]
                annot['visible'] += [np.zeros(16)]
                if trainRef[idx] == 0:  # Test image
                    annot['istrain'] += [2]
                else:   # Training image (something missing in annot)
                    annot['istrain'] += [-1]
    
    
with h5py.File('lsp_mpii.h5','w') as f:
    for k in keys:
        print k
        f[k] = np.array(annot[k])
