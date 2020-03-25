# Code copied and slightly modified from:
# https://github.com/CasiaFan/Dataset_to_VOC_converter/blob/master/anno_coco2voc.py
#
# Most modifications are hlighlighted by the keyword "EDITED".


import argparse, json
import cytoolz
from lxml import etree, objectify
import os, re
import urllib.request, pdb
from urllib.parse import urlparse


def instance2xml_base(anno, download_images):
    # EDITED - make coco_url optional since only used when downloading the images
    if 'coco_url' not in anno:
        if 'url' in anno:
            anno['coco_url'] = anno['url']
        elif not download_images:
            anno['coco_url'] = "not used anywhere in code"
        else:
            raise Exception("Annotation has to contain a 'url' or 'coco_url' field to download the image.")

    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('VOC2014_instance/{}'.format(anno['category_id'])),
        E.filename(anno['file_name']),
        E.source(
            E.database('MS COCO 2014'),
            E.annotation('MS COCO 2014'),
            E.image('Flickr'),
            E.url(anno['coco_url'])
        ),
        E.size(
            E.width(anno['width']),
            E.height(anno['height']),
            E.depth(3)
        ),
        E.segmented(0),
    )
    return anno_tree


def instance2xml_bbox(anno, bbox_type='xyxy'):
    """bbox_type: xyxy (xmin, ymin, xmax, ymax); xywh (xmin, ymin, width, height)"""
    assert bbox_type in ['xyxy', 'xywh']
    if bbox_type == 'xyxy':
        xmin, ymin, w, h = anno['bbox']
        xmax = xmin+w
        ymax = ymin+h
    else:
        xmin, ymin, xmax, ymax = anno['bbox']
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.object(
        E.name(anno['category_id']),
        E.bndbox(
            E.xmin(xmin),
            E.ymin(ymin),
            E.xmax(xmax),
            E.ymax(ymax)
        ),
        E.difficult(anno['iscrowd'])
    )
    return anno_tree


def parse_instance(content, outdir, download_images = False):
    categories = {d['id']: d['name'] for d in content['categories']}

    # EDITED - make sure image_id is of type int (and not of type string)
    for i in range(len(content['annotations'])):
        content['annotations'][i]['image_id'] = int(content['annotations'][i]['image_id'])

    # EDITED - save all annotation .xml files into same sub-directory
    anno_dir = os.path.join(outdir, "annotations")
    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)

    # EDITED - download images
    if download_images:
        im_dir = os.path.join(outdir, "images")
        if not os.path.exists(im_dir):
            os.makedirs(im_dir)

        for index, obj in enumerate(content['images']):
            print(f"Downloading image {index} of {len(content['images'])} from: {obj['coco_url']}")

            # Update 'filename' field to be a (local) filename and not a url
            im_local_filename = os.path.splitext(os.path.basename(obj['file_name']))[0] + ".jpg"
            obj['file_name'] = im_local_filename

            # download image
            dst_path = os.path.join(im_dir, im_local_filename)
            urllib.request.urlretrieve(obj['coco_url'], dst_path)

    # merge images and annotations: id in images vs image_id in annotations
    merged_info_list = list(map(cytoolz.merge, cytoolz.join('id', content['images'], 'image_id', content['annotations'])))
    
    # convert category id to name
    for instance in merged_info_list:
        assert 'category_id' in instance, f"WARNING: annotation error: image {instance['file_name']} has a rectangle without a 'category_id' field."
        instance['category_id'] = categories[instance['category_id']]

    # group by filename to pool all bbox in same file
    img_filenames = {}
    names_groups = cytoolz.groupby('file_name', merged_info_list).items()
    for index, (name, groups) in enumerate(names_groups):
        print(f"Converting annotations for image {index} of {len(names_groups)}: {name}")
        assert not name.lower().startswith(("http:","https:")), "Image seems to be a url rather than local. Need to set 'download_images' = False"

        anno_tree = instance2xml_base(groups[0], download_images)
        # if one file have multiple different objects, save it in each category sub-directory
        filenames = []
        for group in groups:
            filename = os.path.splitext(name)[0] + ".xml"

            # EDITED - save all annotations in single folder, rather than separate folders for each object 
            #filenames.append(os.path.join(outdir, re.sub(" ", "_", group['category_id']), filename)) 
            filenames.append(os.path.join(anno_dir, filename))

            anno_tree.append(instance2xml_bbox(group, bbox_type='xyxy'))

        for filename in filenames:
            etree.ElementTree(anno_tree).write(filename, pretty_print=True)


def keypoints2xml_base(anno):
    annotation = etree.Element("annotation")
    etree.SubElement(annotation, "folder").text = "VOC2014_keypoints"
    etree.SubElement(annotation, "filename").text = anno['file_name']
    source = etree.SubElement(annotation, "source")
    etree.SubElement(source, "database").text = "MS COCO 2014"
    etree.SubElement(source, "annotation").text = "MS COCO 2014"
    etree.SubElement(source, "image").text = "Flickr"
    etree.SubElement(source, "url").text = anno['coco_url']
    size = etree.SubElement(annotation, "size")
    etree.SubElement(size, "width").text = str(anno["width"])
    etree.SubElement(size, "height").text = str(anno["height"])
    etree.SubElement(size, "depth").text = '3'
    etree.SubElement(annotation, "segmented").text = '0'
    return annotation


def keypoints2xml_object(anno, xmltree, keypoints_dict, bbox_type='xyxy'):
    assert bbox_type in ['xyxy', 'xywh']
    if bbox_type == 'xyxy':
        xmin, ymin, w, h = anno['bbox']
        xmax = xmin+w
        ymax = ymin+h
    else:
        xmin, ymin, xmax, ymax = anno['bbox']
    key_object = etree.SubElement(xmltree, "object")
    etree.SubElement(key_object, "name").text = anno['category_id']
    bndbox = etree.SubElement(key_object, "bndbox")
    etree.SubElement(bndbox, "xmin").text = str(xmin)
    etree.SubElement(bndbox, "ymin").text = str(ymin)
    etree.SubElement(bndbox, "xmax").text = str(xmax)
    etree.SubElement(bndbox, "ymax").text = str(ymax)
    etree.SubElement(key_object, "difficult").text = '0'
    keypoints = etree.SubElement(key_object, "keypoints")
    for i in range(0, len(keypoints_dict)):
        keypoint = etree.SubElement(keypoints, keypoints_dict[i+1])
        etree.SubElement(keypoint, "x").text = str(anno['keypoints'][i*3])
        etree.SubElement(keypoint, "y").text = str(anno['keypoints'][i*3+1])
        etree.SubElement(keypoint, "v").text = str(anno['keypoints'][i*3+2])
    return xmltree


def parse_keypoints(content, outdir):
    keypoints = dict(zip(range(1, len(content['categories'][0]['keypoints'])+1), content['categories'][0]['keypoints']))
    # merge images and annotations: id in images vs image_id in annotations
    merged_info_list = map(cytoolz.merge, cytoolz.join('id', content['images'], 'image_id', content['annotations']))
    # convert category name to person
    for keypoint in merged_info_list:
        keypoint['category_id'] = "person"
    # group by filename to pool all bbox and keypoint in same file
    for name, groups in cytoolz.groupby('file_name', merged_info_list).items():
        filename = os.path.join(outdir, os.path.splitext(name)[0]+".xml")
        anno_tree = keypoints2xml_base(groups[0])
        for group in groups:
            anno_tree = keypoints2xml_object(group, anno_tree, keypoints, bbox_type="xyxy")
        doc = etree.ElementTree(anno_tree)
        doc.write(open(filename, "w"), pretty_print=True)
        print("Formating keypoints xml file {} done!".format(name))


def coco2voc_main(anno_file, output_dir, anno_type, download_images = False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    content = json.load(open(anno_file, 'r'))
    
    if anno_type == 'instance':
        # EDITED - save all annotations in single folder, rather than separate folders for each object 
        # make subdirectories
        # sub_dirs = [re.sub(" ", "_", cate['name']) for cate in content['categories']]   #EDITED
        # for sub_dir in sub_dirs:
        #     sub_dir = os.path.join(output_dir, str(sub_dir))
        #     if not os.path.exists(sub_dir):
        #         os.makedirs(sub_dir)
        parse_instance(content, output_dir, download_images)
    elif anno_type == 'keypoint':
        parse_keypoints(content, output_dir)
    else:
        error
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_file", help="annotation file for object instance/keypoint")
    parser.add_argument("--type", type=str, help="object instance or keypoint", choices=['instance', 'keypoint'])
    parser.add_argument("--output_dir", help="output directory for voc annotation xml file")
    args = parser.parse_args()
    main(args.anno_file, args.output_dir, args.type)
