# Labelbox Annotation Conversion #

* [PASCAL VOC Annotation](#pascal-voc-annotation)
* [Mask Annotation](#mask-annotation)
  + [Mask Annotation Steps](#mask-annotation-steps)
  + [Mask Meta Data Structure](#mask-meta-data-structure)
  + [Convert Mask Annotation Data](#convert-mask-annotation-data)
    - [Use Exporters from LabelBox](#use-exporters-from-labelbox)
    - [Manually Merge in Segmentation Masks](#manually-merge-in-segmentation-masks)
* [Keypoint Annotation](#keypoint-annotation)
  + [Keypoint Annotation Steps](#keypoint-annotation-steps)
  + [Keypoint Meta Data Structure](#keypoint-meta-data-structure)
  + [Convert Keypoint Annotation Data](#convert-keypoint-annotation-data)


## PASCAL VOC Annotation ##

The annotation files in
[odFridgeObjects](https://cvbp.blob.core.windows.net/public/datasets/object_detection/odFridgeObjects.zip)
are in the format of PASCAL VOC.

```console
$ # Download and unzip original dataset
$ URL='https://cvbp.blob.core.windows.net/public/datasets/object_detection/odFridgeObjects.zip'
$ ZIPBALL="${URL##*/}"; ZIP_DIR="${ZIPBALL%.zip}"; ANNO_DIR="${ZIP_DIR}/annotations"
$ wget -O ${ZIPBALL} ${URL}
$ unzip ${ZIPBALL}
$
$ cat ${ANNO_DIR}/42.xml
<annotation>
	<folder>images</folder>
	<filename>42.jpg</filename>
	<path>../images/42.jpg</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>499</width>
		<height>666</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>milk_bottle</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>65</xmin>
			<ymin>264</ymin>
			<xmax>184</xmax>
			<ymax>545</ymax>
		</bndbox>
	</object>
	<object>
		<name>carton</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>144</xmin>
			<ymin>308</ymin>
			<xmax>408</xmax>
			<ymax>508</ymax>
		</bndbox>
	</object>
	<object>
		<name>water_bottle</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>337</xmin>
			<ymin>175</ymin>
			<xmax>428</xmax>
			<ymax>404</ymax>
		</bndbox>
	</object>
</annotation>

```


## Mask Annotation ##

### Mask Annotation Steps ###

The general steps for mask annotation at Labelbox:

1. New project
1. Add dataset
1. Add objects and classification if any
1. Start labeling
1. Add to back if the objects to be annotated are occluded by already
   annotated objects
1. Export and download JSON file
1. Convert donwloaded JSON file into desired format


### Mask Meta Data Structure ###

The data structure of the export JSON file from Labelbox:

```pycon
>>> import json
>>> labelbox_json = 'masks/export-2019-11-18T02_23_40.854Z.json'
>>> with open(labelbox_json) as f:
...     annos = json.load(f)
...
>>> type(annos)
<class 'list'>
>>> len(annos)
128
>>> from pprint import pprint
>>> pprint(annos[0])
{'Agreement': None,
 'Benchmark Agreement': None,
 'Benchmark ID': None,
 'Benchmark Reference ID': None,
 'Created At': '2019-10-09T03:37:48.000Z',
 'Created By': 'simonyansenzhao@gmail.com',
 'DataRow ID': 'ck1ipfp7trezp0cwb25hsbo47',
 'Dataset Name': 'odFridgeObjects',
 'External ID': '117.jpg',
 'ID': 'ck1iq31v1qqht0863h6xwnr1a',
 'Label': {'classifications': [{'answers': [{'featureId': 'ck1iq2wh43gwg0a46wp0dxfbp',
                                             'schemaId': 'ck1ipz4v5s5rb0701nlup2h7r',
                                             'title': 'aa',
                                             'value': 'aa'},
                                            {'featureId': 'ck1iq2x2qv63y0721m09e851f',
                                             'schemaId': 'ck1ipz4v5s5rc0701d4eneyiq',
                                             'title': 'bb',
                                             'value': 'bb'}],
                                'featureId': 'ck1iq2wf7s7kw0701wm0o8gsk',
                                'schemaId': 'ck1ipz4wlppom083840dab0ss',
                                'title': 'bottle',
                                'value': 'bottle'}],
           'objects': [{'classifications': [{'answer': {'featureId': 'ck1iucj8cu64m0701h1hm8lry',
                                                        'schemaId': 'ck1itgreusaeq08635ay6myxl',
                                                        'title': '0',
                                                        'value': '0'},
                                             'featureId': 'ck1iucj5mv00p09443mel4w2v',
                                             'schemaId': 'ck1itgrie6etz0848dh3wzprg',
                                             'title': 'iscrowd',
                                             'value': 'iscrowd'}],
                        'color': '#00D4FF',
                        'featureId': 'ck1iu6m3suwmo0944zoufayto',
                        'instanceURI': 'https://api.labelbox.com/masks/ck1iphg4xsqhe0944bbbiwrak/ck1ipbug6qe8h086379jggd93/ck1ipfp7trezp0cwb25hsbo47/0?feature=ck1iu6m3suwmo0944zoufayto&token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjazFpcGJ1ZzZxZThoMDg2Mzc5amdnZDkzIiwib3JnYW5pemF0aW9uSWQiOiJjazFpcGJ1ZmF1dTRmMDcyMTBwNG1jOTJ4IiwiaWF0IjoxNTczMjAwOTc0LCJleHAiOjE1NzU3OTI5NzR9.YVRqmrcep0WDlokfoxw-qSJhJ3pnMynll0PopzI6WAg',
                        'schemaId': 'ck1ipz4v5s5rd0701j2mfc4ii',
                        'title': 'water_bottle',
                        'value': 'water_bottle'},
                       {'classifications': [{'answer': {'featureId': 'ck1iup8ohxd6u0721smk9e9xq',
                                                        'schemaId': 'ck1itgreusaes0863j1nzm7cm',
                                                        'title': '0',
                                                        'value': '0'},
                                             'featureId': 'ck1iup8m1v6tj09444wrkpcoz',
                                             'schemaId': 'ck1itgrie6eu00848axtbt9eh',
                                             'title': 'iscrowd',
                                             'value': 'iscrowd'}],
                        'color': '#00FFFF',
                        'featureId': 'ck1iuonmvryt608388vlq6t9z',
                        'instanceURI': 'https://api.labelbox.com/masks/ck1iphg4xsqhe0944bbbiwrak/ck1ipbug6qe8h086379jggd93/ck1ipfp7trezp0cwb25hsbo47/0?feature=ck1iuonmvryt608388vlq6t9z&token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjazFpcGJ1ZzZxZThoMDg2Mzc5amdnZDkzIiwib3JnYW5pemF0aW9uSWQiOiJjazFpcGJ1ZmF1dTRmMDcyMTBwNG1jOTJ4IiwiaWF0IjoxNTczMjAwOTc0LCJleHAiOjE1NzU3OTI5NzR9.YVRqmrcep0WDlokfoxw-qSJhJ3pnMynll0PopzI6WAg',
                        'schemaId': 'ck1ipz4v5s5re0701sojrveb3',
                        'title': 'milk_bottle',
                        'value': 'milk_bottle'}]},
 'Labeled Data': 'https://storage.labelbox.com/ck1ipbufauu4f07210p4mc92x%2Fa03eeb05-044b-6e37-b3f7-58d748d4418a-117.jpg?Expires=1574410574113&KeyName=labelbox-assets-key-1&Signature=8dIqH79twHJB4k-skYdk2y33fuM',
 'Project Name': 'odFridgeObjects',
 'Reviews': [],
 'Seconds to Label': 3950.709,
 'Updated At': '2019-10-12T01:54:16.000Z',
 'View Label': 'https://editor.labelbox.com?project=ck1iphg4xsqhe0944bbbiwrak&label=ck1iq31v1qqht0863h6xwnr1a'}
```

`annos` is a list of `Dict` where each `Dict` is the meta data for an
image.  Key fields include:
* **`annos[n]['External ID']`**: Original image file name
* `annos[n]['Labeled Data']`: URL of the original image
* `annos[n]['View Label']`: URL of the image with labels or masks
* `annos[n]['Label']`: Dict.  Meta data of all annotations of the
  image
* `annos[n]['Label']['objects']`: List.  Meta data of all objects of
  the image.
* `annos[n]['Label']['objects'][0]['value']`: Object name (category)
* **`annos[n]['Label']['objects'][0]['instanceURI']`**: URL of the
  binary mask of the object, with 0 as background, 255 as the object.


### Convert Mask Annotation Data ###

LabelBox provides Python functions for converting the JSON file from
LabelBox to PASCAL VOC or COCO format.  However, the functions are not
easy-to-use.  So we provide our own scripts below.

Take the
[`odFridgeObjects`](https://cvbp.blob.core.windows.net/public/datasets/object_detection/odFridgeObjects.zip)
dataset as an example.  Here the XML annotations are in the
`odFridgeObjects/annotations` folder and the original images are in
the `odFridgeObjects/images` folder.  For an arbitratry image
`odFridgeObjects/images/xyz.jpg`, its corresponding XML annotation
file is `odFridgeObjects/annotations/xyz.xml`.

Because the missing parts are the masks annotated in LabelBox, the
only thing we need to do is to combine all binary masks
(`[obj['instanceURI'] for obj in annos[0]['Label']['objects']]`) of
individual objects from an image (`annos[0]['External ID']`) into a
single mask image (`annos[0]['External ID'][:-4] + '.png'`) in a
directory called `segmentation-masks`.

Note the objects exported from LabelBox may not be in the order
described in the PASCAL POC XML file, so reordering is needed.

Below are the steps to do the conversion.

```console
$ # generate conversion script `extract_masks.py`
$ cat > extract_masks.py <<<EOF
import matplotlib.pyplot as plt     # image display, verification
import json                         # read LabelBox JSON file
import numpy as np                  # image processing
import shutil                       # file manipulation
import sys                          # Python command line arguments
import urllib.request               # download
import xml.etree.ElementTree as ET  # read PASCAL VOC XML annotation

from PIL import Image               # read image
from pathlib import Path            # path manipulation

labelbox_json = sys.argv[1]            # LabelBox JSON file
old_dir = Path(sys.argv[2])            # path to original dataset
new_dir = Path(sys.argv[3])            # path to masked dataset

old_img_dir = old_dir / 'images'        # image folder
old_anno_dir = old_dir / 'annotations'  # annotation folder

new_img_dir = new_dir / 'images'
new_anno_dir = new_dir / 'annotations'
new_mask_dir = new_dir / 'segmentation-masks'  # mask folder

# create directories for annotated dataset
new_img_dir.mkdir(parents=True, exist_ok=True)
new_anno_dir.mkdir(parents=True, exist_ok=True)
new_mask_dir.mkdir(parents=True, exist_ok=True)

# read exported LabelBox annotation JSON file
with open(labelbox_json) as f:
    annos = json.load(f)

# process one image per iteration
for anno in annos:
    # get related file paths
    im_name = anno['External ID']      # image file name
    anno_name = im_name[:-4] + '.xml'  # annotation file name
    mask_name = im_name[:-4] + '.png'  # mask file name

    print('Processing image: {}'.format(im_name))

    old_img = old_img_dir / im_name
    old_anno = old_anno_dir / anno_name
    
    new_img = new_img_dir / im_name
    new_anno = new_anno_dir / anno_name
    new_mask = new_mask_dir / mask_name

    # copy original image and annotation file
    shutil.copy(old_img, new_img)
    shutil.copy(old_anno, new_anno)

    # read mask images
    mask_urls = [obj['instanceURI'] for obj in anno['Label']['objects']]
    labels = [obj['value'] for obj in anno['Label']['objects']]
    binary_masks = np.array([
        np.array(Image.open(urllib.request.urlopen(url)))[..., 0] == 255 for
        url in mask_urls
    ])

    # rearrange masks with regard to annotation
    tree = ET.parse(new_anno)
    root = tree.getroot()
    rects = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        bnd_box = obj.find('bndbox')
        left = int(bnd_box.find('xmin').text)
        top = int(bnd_box.find('ymin').text)
        right = int(bnd_box.find('xmax').text)
        bottom = int(bnd_box.find('ymax').text)
        rects.append((label, left, top, right, bottom))

    assert len(rects) == len(binary_masks)
    matches = []
    # find matched binary mask and annotation
    for label, left, top, right, bottom in rects:
        match = 0
        min_overlap = binary_masks.shape[1]*binary_masks.shape[2]
        for i, bmask in enumerate(binary_masks):
            bmask_out = bmask.copy()
            bmask_out[top:(bottom+1), left:(right+1)] = False
            non_overlap = np.sum(bmask_out)
            if non_overlap < min_overlap:
                match = i
                min_overlap = non_overlap
        assert label == labels[match], '{}: {}'.format(label, labels[match])
        matches.append(match)

    assert len(set(matches)) == len(matches), '{}: {}'.format(len(set(matches)), len(matches))

    if [i for i in range(len(matches))] != matches:
        print('    Reorder happend!')
    
    binary_masks = binary_masks[matches]
    
    # merge binary masks
    obj_values = np.arange(len(binary_masks)) + 1
    labeled_masks = binary_masks * obj_values[:, None, None]
    print('    {}'.format(labeled_masks.shape))
    mask = np.max(labeled_masks, axis=0).astype(np.uint8)

    # save mask image
    Image.fromarray(mask, mode='L').save(new_mask)

    # # view the mask image
    # plt.imshow(Image.open(new_mask))
    # plt.show()

EOF
$
$ # Download and unzip original dataset
$ URL='https://cvbp.blob.core.windows.net/public/datasets/object_detection/odFridgeObjects.zip'
$ ZIPBALL="${URL##*/}"; ZIP_DIR="${ZIPBALL%.zip}"; RES_DIR="${ZIP_DIR}Mask"
$ LABELBOX_JSON='masks/export-2019-11-18T02_23_40.854Z.json'
$ wget -O ${ZIPBALL} ${URL}
$ unzip ${ZIPBALL}
$
$ # set up envrionment
$ conda activate cvbp
(cvbp) $ 
(cvbp) $ # conversion
(cvbp) $ python extract_masks.py ${LABELBOX_JSON} ${ZIP_DIR} ${RES_DIR}
(cvbp) $ zip -r ${RES_DIR}.zip ${RES_DIR}
```


## Keypoint Annotation ##

### Keypoint Annotation Steps ###

1. New project
1. Add dataset
1. Add point labels if any
   * Need to prefix label names with category names.  For example,
     instead of `left_ear`, we should use `person_left_ear`
1. Start labeling
1. Export and download JSON file
1. Convert donwloaded JSON file into desired format 


### Keypoint Meta Data Structure ###

```pycon
>>> import json
>>> labelbox_json = "keypoints/export-2019-11-28T08_27_44.729Z.json"
>>> with open(labelbox_json) as f:
...     annos = json.load(f)
... 
>>> type(annos)
<class 'list'>
>>> len(annos)
30
>>> from pprint import pprint
>>> pprint(annos[0])
{'Agreement': None,
 'Benchmark Agreement': None,
 'Benchmark ID': None,
 'Benchmark Reference ID': None,
 'Created At': '2019-11-20T06:48:16.000Z',
 'Created By': 'simonyansenzhao@gmail.com',
 'DataRow ID': 'ck1ipfp7mrep10cwb09n8copx',
 'Dataset Name': 'odFridgeObjects',
 'External ID': '21.jpg',
 'ID': 'ck36xdrzryw3r0721zbe83tkw',
 'Label': {'carton_left_back_bottom': [{'geometry': {'x': 217, 'y': 277}}],
           'carton_left_back_shoulder': [{'geometry': {'x': 410, 'y': 340}}],
           'carton_left_collar': [{'geometry': {'x': 416, 'y': 367}}],
           'carton_left_front_bottom': [{'geometry': {'x': 161, 'y': 299}}],
           'carton_left_front_shoulder': [{'geometry': {'x': 359, 'y': 375}}],
           'carton_left_top': [{'geometry': {'x': 438, 'y': 379}}],
           'carton_lid': [{'geometry': {'x': 392, 'y': 427}}],
           'carton_right_collar': [{'geometry': {'x': 398, 'y': 450}}],
           'carton_right_front_bottom': [{'geometry': {'x': 166, 'y': 371}}],
           'carton_right_front_shoulder': [{'geometry': {'x': 350, 'y': 462}}],
           'carton_right_top': [{'geometry': {'x': 424, 'y': 455}}],
           'water_bottle_lid_left_bottom': [{'geometry': {'x': 243, 'y': 444}}],
           'water_bottle_lid_left_top': [{'geometry': {'x': 266, 'y': 456}}],
           'water_bottle_lid_right_bottom': [{'geometry': {'x': 220,
                                                           'y': 499}}],
           'water_bottle_lid_right_top': [{'geometry': {'x': 243, 'y': 511}}],
           'water_bottle_wrapper_left_bottom': [{'geometry': {'x': 77,
                                                              'y': 344}}],
           'water_bottle_wrapper_left_top': [{'geometry': {'x': 161,
                                                           'y': 379}}],
           'water_bottle_wrapper_right_bottom': [{'geometry': {'x': 30,
                                                               'y': 424}}],
           'water_bottle_wrapper_right_top': [{'geometry': {'x': 120,
                                                            'y': 477}}]},
 'Labeled Data': 'https://storage.labelbox.com/ck1ipbufauu4f07210p4mc92x%2F047eb583-3075-c180-80cc-5748106f5ce6-21.jpg?Expires=1575512702598&KeyName=labelbox-assets-key-1&Signature=8DB1bsgkYWlld3QeVWxd6dWwOWs',
 'Project Name': 'odFridgeObjectsKeypoints',
 'Reviews': [],
 'Seconds to Label': 443.956,
 'Updated At': '2019-11-20T06:48:16.000Z',
 'View Label': 'https://image-segmentation-v4.labelbox.com?project=ck36v24hrxor407215xlodfo9&label=ck36xdrzryw3r0721zbe83tkw'}
```

`annos` is a list of `Dict` where each `Dict` is the meta data for an
image.  Key fields include:
* **`annos[n]['External ID']`**: Original image file name
* `annos[n]['Labeled Data']`: URL of the original image
* `annos[n]['View Label']`: URL of the image with labels or masks
* `annos[n]['Label']`: Dict.  Meta data of all annotations of the
  image.  Its keys are the labels of keypoints, and its values are the
  coordinates.
* **`annos[n]['Label']['xxx'][0]['geometry']['x']`**: The x coordinate
  of the label `xxx`.
* **`annos[n]['Label']['xxx'][0]['geometry']['y']`**: The y coordinate
  of the label `xxx`.

**NOTE** that things become tricky when there are multiple instances
of the same category exist in an image.  But for now in the
odFridgeObjects dataset, no more than one instance of a category
exists in an image.  In addition, there is no natural way of
specifying a point belongs to a label.  Therefore, for example, we use
the prefix `carton_` to indicate the point labeled
`carton_left_back_bottom` is a point that belongs to a carton.


### Convert Keypoint Annotation Data ###

```console
$ # generate conversion script `extract_keypoints.py`
$ cat > extract_keypoints.py <<<EOF
import json                         # read LabelBox JSON file
from pathlib import Path            # path manipulation
import shutil                       # file manipulation
import sys                          # Python command line arguments
import xml.etree.ElementTree as ET  # read PASCAL VOC XML annotation


#  indent XML output
# * http://effbot.org/zone/element-lib.htm#prettyprint
# * https://stackoverflow.com/a/4590052
def indent(elem, level=0):
    spaces = '        '
    i = '\n' + level * spaces
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + spaces
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


# get data and keypoint annotation JSON file from command line
labelbox_json = sys.argv[1]               # LabelBox JSON file
old_dir = Path(sys.argv[2])               # path to original dataset
new_dir = Path(sys.argv[3])               # path to keypoint dataset

old_img_dir = old_dir / 'images'        # original image folder
old_anno_dir = old_dir / 'annotations'  # original annotation folder

new_img_dir = new_dir / 'images'        # keypoint image folder
new_anno_dir = new_dir / 'annotations'  # keypoint annotation folder

# create directories for annotated dataset
new_img_dir.mkdir(parents=True, exist_ok=True)
new_anno_dir.mkdir(parents=True, exist_ok=True)

# read exported LabelBox annotation JSON file
with open(labelbox_json) as f:
    annos = json.load(f)

# process one image keypoints annotation per iteration
for anno in annos:
    # get related file paths
    im_name = anno['External ID']      # image file name
    anno_name = im_name[:-4] + '.xml'  # annotation file name

    print('Processing image: {}'.format(im_name))

    old_img = old_img_dir / im_name
    old_anno = old_anno_dir / anno_name

    new_img = new_img_dir / im_name
    new_anno = new_anno_dir / anno_name

    # copy original image
    shutil.copy(old_img, new_img)

    # add keypoints annotation into PASCAL VOC XML file
    kps_annos = anno['Label']
    tree = ET.parse(old_anno)
    root = tree.getroot()
    for obj in root.findall('object'):
        prefix = obj.find('name').text + '_'
        kps = ET.SubElement(obj, 'keypoints')  # add 'keypoints' node for current object
        for k in kps_annos.keys():
            if k.startswith(prefix):
                pt = ET.SubElement(kps, k[len(prefix):])  # add keypoint into 'keypoints' node
                x = ET.SubElement(pt, 'x')  # add x coordinate
                y = ET.SubElement(pt, 'y')  # add y coordinate
                geo = kps_annos[k][0]['geometry']
                x.text = str(geo['x'])
                y.text = str(geo['y'])

    # format XML
    indent(root)

    # write modified annotation file
    tree.write(new_anno)

EOF
$
$ # Download and unzip original dataset
$ URL='https://cvbp.blob.core.windows.net/public/datasets/object_detection/odFridgeObjects.zip'
$ ZIPBALL="${URL##*/}"; ZIP_DIR="${ZIPBALL%.zip}"; RES_DIR="${ZIP_DIR}Keypoint"
$ LABELBOX_JSON='keypoints/export-2019-11-28T08_27_44.729Z.json'
$ wget -O ${ZIPBALL} ${URL}
$ unzip ${ZIPBALL}
$
$ # set up envrionment
$ conda activate cvbp
(cvbp) $ 
(cvbp) $ # conversion
(cvbp) $ python extract_keypoints.py ${LABELBOX_JSON} ${ZIP_DIR} ${RES_DIR}
(cvbp) $ zip -r ${RES_DIR}.zip ${RES_DIR}
```
