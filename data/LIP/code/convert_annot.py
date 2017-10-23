import h5py
import numpy as np
import sys
import cv2

keys = ['imgname','center','scale','part','visible','istrain']
annot = {k:[] for k in keys}


train_fn = "lip_train_set.csv"
train_dl = np.array([l.strip() for l in open(train_fn).readlines()])
val_fn = "lip_val_set.csv"
val_dl = np.array([l.strip() for l in open(val_fn).readlines()])
test_fn = "pose_result.csv"
test_dl = np.array([l.strip() for l in open(test_fn).readlines()])

for i, line in enumerate(train_dl):
	print i
	datum = line.split(',')

	joints = np.asarray([float(p) for p in datum[1:49]])
	joints = joints.reshape((len(joints) / 3, 3))
	#joints = joints[[8,7,6,9,10,11,15,14,13,12,2,1,0,3,4,5],:]
	visible = joints[:,2].copy()
	coords = joints[:,:2]

	img = cv2.imread('train_images/'+datum[0]) 
	h, w, _ = img.shape
	s = 0.8*max(w,h)/200
	c = np.array([w/2,h/2])

	#jointsc = jointsc.astype(np.int32)
	#x, y, w, h = cv2.boundingRect(np.asarray([jointsc.tolist()]))
	#c = np.array([x+w/2, y+w/2])
	#c = np.array([float(datum[-2]),float(datum[-1])])

	imgnameRef = 'train_images/' + datum[0] + '*'
	imgname = np.zeros(60)
	refname = str(imgnameRef)
	for i in range(len(refname)): imgname[i] = ord(refname[i])
	annot['imgname'] += [imgname]
	annot['center'] += [c]
	annot['scale'] += [s]            
	annot['part'] += [coords]
	annot['visible'] += [visible]
	annot['istrain'] += [1]
        
for i, line in enumerate(val_dl):
    datum = line.split(',')

    joints = np.asarray([float(p) for p in datum[1:49]])
    joints = joints.reshape((len(joints) / 3, 3))
    #joints = joints[[8,7,6,9,10,11,15,14,13,12,2,1,0,3,4,5],:]
    visible = joints[:,2].copy()
    coords = joints[:,:2]

    img = cv2.imread('val_images/'+datum[0]) 
    h, w, _ = img.shape
    s = 0.8*max(w,h)/200
    c = np.array([w/2,h/2])

    #jointsc = jointsc.astype(np.int32)
    #x, y, w, h = cv2.boundingRect(np.asarray([jointsc.tolist()]))
    #c = np.array([x+w/2, y+w/2])
    #c = np.array([w/2,h/2])

    imgnameRef = 'val_images/' + datum[0] + '*'
    imgname = np.zeros(60)
    refname = str(imgnameRef)
    for i in range(len(refname)): imgname[i] = ord(refname[i])
    annot['imgname'] += [imgname]
    annot['center'] += [c]
    annot['scale'] += [s]                  
    annot['part'] += [coords]
    annot['visible'] += [visible]
    annot['istrain'] += [2]

for i, line in enumerate(test_dl):
    print 'testing_images/'+line
    
    coords = np.zeros((16,2))
    visible = np.zeros(16)

    img = cv2.imread('testing_images/'+line) 
    h, w, _ = img.shape
    s = 0.8*max(w,h)/200
    c = np.array([w/2,h/2])

    #jointsc = jointsc.astype(np.int32)
    #x, y, w, h = cv2.boundingRect(np.asarray([jointsc.tolist()]))
    #c = np.array([x+w/2, y+w/2])
    #c = np.array([w/2,h/2])

    imgnameRef = 'testing_images/' + line + '*'
    imgname = np.zeros(60)
    refname = str(imgnameRef)
    for i in range(len(refname)): imgname[i] = ord(refname[i])
    #print imgname
    annot['imgname'] += [imgname]
    annot['center'] += [c]
    annot['scale'] += [s]                
    annot['part'] += [coords]
    annot['visible'] += [visible]
    annot['istrain'] += [0]
	
with h5py.File('lip.h5','w') as f:
    f.attrs['name'] = 'lip'
    for k in keys:
        print k
        f[k] = np.array(annot[k])
