#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
from scipy import ndimage as ndi
from skimage import feature
from skimage import morphology
import cv2
from plantcv import plantcv as pcv


def options():
    parser = argparse.ArgumentParser(description="ENCODE PlantCV workflow.")
    parser.add_argument("-i", "--image", help="Input image file.", required=True)
    parser.add_argument("-d", "--debug", help="Turn on debug, prints intermediate images.", default=None)
    parser.add_argument("-w", "--writeimg", help="write out images.", default=False, action="store_true")
    parser.add_argument("-r", "--result", help="result file.", required=False)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
    parser.add_argument("-n", "--bkg", help="JSON config file for background images.", required=True)
    parser.add_argument("-p", "--pdf", help="PDF from Naive-Bayes.", required=True)
    args = parser.parse_args()
    return args


def main():
    # create options object for argument parsing
    args = options()
    # set device
    device = 0
    # set debug
    pcv.params.debug = args.debug

    outfile = False
    if args.writeimg:
        outfile = os.path.join(args.outdir, os.path.basename(args.image)[:-4])

    # read in image
    img, path, filename = pcv.readimage(filename=args.image, debug=args.debug)

    # read in a background image for each zoom level
    config_file = open(args.bkg, 'r')
    config = json.load(config_file)
    config_file.close()
    if "z1500" in args.image:
        bkg_image = config["z1500"]
    elif "z2500" in args.image:
        bkg_image = config["z2500"]
    else:
        pcv.fatal_error("Image {0} has an unsupported zoom level.".format(args.image))

    bkg, bkg_path, bkg_filename = pcv.readimage(filename=bkg_image, debug=args.debug)

    # Detect edges in the background image
    device, bkg_sat = pcv.rgb2gray_hsv(img=bkg, channel="s", device=device, debug=args.debug)
    device += 1
    bkg_edges = feature.canny(bkg_sat)
    if args.debug == "print":
        pcv.print_image(img=bkg_edges, filename=str(device) + '_background_edges.png')
    elif args.debug == "plot":
        pcv.plot_image(img=bkg_edges, cmap="gray")

    # Close background edge contours
    bkg_edges_closed = ndi.binary_closing(bkg_edges)
    device += 1
    if args.debug == "print":
        pcv.print_image(img=bkg_edges_closed, filename=str(device) + '_closed_background_edges.png')
    elif args.debug == "plot":
        pcv.plot_image(img=bkg_edges_closed, cmap="gray")

    # Fill in closed contours in background
    bkg_fill_contours = ndi.binary_fill_holes(bkg_edges_closed)
    device += 1
    if args.debug == "print":
        pcv.print_image(img=bkg_fill_contours, filename=str(device) + '_filled_background_edges.png')
    elif args.debug == "plot":
        pcv.plot_image(img=bkg_fill_contours, cmap="gray")

    # Naive Bayes image classification/segmentation
    device, mask = pcv.naive_bayes_classifier(img=img, pdf_file=args.pdf, device=device, debug=args.debug)

    # Do a light cleaning of the plant mask to remove small objects
    cleaned = morphology.remove_small_objects(mask["plant"].astype(bool), 2)
    device += 1
    if args.debug == "print":
        pcv.print_image(img=cleaned, filename=str(device) + '_cleaned_mask.png')
    elif args.debug == "plot":
        pcv.plot_image(img=cleaned, cmap="gray")

    # Convert the input image to a saturation channel grayscale image
    device, sat = pcv.rgb2gray_hsv(img=img, channel="s", device=device, debug=args.debug)

    # Detect edges in the saturation image
    edges = feature.canny(sat)
    device += 1
    if args.debug == "print":
        pcv.print_image(img=edges, filename=str(device) + '_plant_edges.png')
    elif args.debug == "plot":
        pcv.plot_image(img=edges, cmap="gray")

    # Combine pixels that are in both foreground edges and the filled background edges
    device, combined_bkg = pcv.logical_and(img1=edges.astype(np.uint8) * 255,
                                           img2=bkg_fill_contours.astype(np.uint8) * 255, device=device,
                                           debug=args.debug)

    # Remove background pixels from the foreground edges
    device += 1
    filtered = np.copy(edges)
    filtered[np.where(combined_bkg == 255)] = False
    if args.debug == "print":
        pcv.print_image(img=filtered, filename=str(device) + '_filtered_edges.png')
    elif args.debug == "plot":
        pcv.plot_image(img=filtered, cmap="gray")

    # Combine the cleaned naive Bayes mask and the filtered foreground edges
    device += 1
    combined = cleaned + filtered
    if args.debug == "print":
        pcv.print_image(img=combined, filename=str(device) + '_combined_foreground.png')
    elif args.debug == "plot":
        pcv.plot_image(img=combined, cmap="gray")

    # Close off broken edges and other incomplete contours
    device += 1
    closed_features = ndi.binary_closing(combined)
    if args.debug == "print":
        pcv.print_image(img=closed_features, filename=str(device) + '_closed_features.png')
    elif args.debug == "plot":
        pcv.plot_image(img=closed_features, cmap="gray")

    # Fill in holes in contours
    device += 1
    fill_contours = ndi.binary_fill_holes(closed_features)
    if args.debug == "print":
        pcv.print_image(img=fill_contours, filename=str(device) + '_filled_contours.png')
    elif args.debug == "plot":
        pcv.plot_image(img=fill_contours, cmap="gray")

    # Use median blur to break horizontal and vertical thin edges (pot edges)
    device += 1
    blurred_img = ndi.median_filter(fill_contours.astype(np.uint8) * 255, (3, 1))
    blurred_img = ndi.median_filter(blurred_img, (1, 3))
    # Remove small objects left behind by blurring
    cleaned2 = morphology.remove_small_objects(blurred_img.astype(bool), 200)
    if args.debug == "print":
        pcv.print_image(img=cleaned2, filename=str(device) + '_cleaned_by_median_blur.png')
    elif args.debug == "plot":
        pcv.plot_image(img=cleaned2, cmap="gray")

    # Define region of interest based on camera zoom level for masking the naive Bayes classified image
    if "z1500" in args.image:
        h = 1000
    elif "z2500" in args.image:
        h = 1050
    else:
        pcv.fatal_error("Image {0} has an unsupported zoom level.".format(args.image))
    roi, roi_hierarchy = pcv.roi.rectangle(x=300, y=150, w=1850, h=h, img=img)

    # Mask the classified image to remove noisy areas prior to finding contours
    side_mask = np.zeros(np.shape(img)[:2], dtype=np.uint8)
    cv2.drawContours(side_mask, roi, -1, (255), -1)
    device, masked_img = pcv.apply_mask(img=cv2.merge([mask["plant"], mask["plant"], mask["plant"]]), mask=side_mask,
                                        mask_color="black", device=device, debug=args.debug)
    # Convert the masked image back to grayscale
    masked_img = masked_img[:, :, 0]
    # Close off contours at the base of the plant
    if "z1500" in args.image:
        pt1 = (1100, 1118)
        pt2 = (1340, 1120)
    elif "z2500" in args.image:
        pt1 = (1020, 1162)
        pt2 = (1390, 1166)
    else:
        pcv.fatal_error("Image {0} has an unsupported zoom level.".format(args.image))
    masked_img = cv2.rectangle(np.copy(masked_img), pt1, pt2, (255), -1)
    closed_mask = ndi.binary_closing(masked_img.astype(bool), iterations=3)

    # Find objects in the masked naive Bayes mask
    # device, objects, obj_hierarchy = pcv.find_objects(img=img, mask=np.copy(masked_img), device=device,
    #                                                   debug=args.debug)
    objects, obj_hierarchy = cv2.findContours(np.copy(closed_mask.astype(np.uint8) * 255), cv2.RETR_CCOMP,
                                              cv2.CHAIN_APPROX_NONE)[-2:]

    # Clean up the combined plant edges/mask image by removing filled in gaps/holes
    device += 1
    cleaned3 = np.copy(cleaned2)
    cleaned3 = cleaned3.astype(np.uint8) * 255
    # Loop over the contours from the naive Bayes mask
    for c, contour in enumerate(objects):
        # Calculate the area of each contour
        # area = cv2.contourArea(contour)
        # If the contour is a hole (i.e. it has no children and it has a parent)
        # And it is not a small hole in a leaf that was not classified
        if obj_hierarchy[0][c][2] == -1 and obj_hierarchy[0][c][3] > -1:
            # Then fill in the contour (hole) black on the cleaned mask
            cv2.drawContours(cleaned3, objects, c, (0), -1, hierarchy=obj_hierarchy)
    if args.debug == "print":
        pcv.print_image(img=cleaned3, filename=str(device) + '_gaps_removed.png')
    elif args.debug == "plot":
        pcv.plot_image(img=cleaned3, cmap="gray")

    # Find contours using the cleaned mask
    device, contours, contour_hierarchy = pcv.find_objects(img=img, mask=np.copy(cleaned3), device=device,
                                                           debug=args.debug)

    # Define region of interest based on camera zoom level for contour filtering
    if "z1500" in args.image:
        h = 940
    elif "z2500" in args.image:
        h = 980
    else:
        pcv.fatal_error("Image {0} has an unsupported zoom level.".format(args.image))
    roi, roi_hierarchy = pcv.roi.rectangle(x=300, y=150, w=1850, h=h, img=img)

    # Filter contours in the region of interest
    device, roi_objects, hierarchy, kept_mask, obj_area = pcv.roi_objects(img=img, roi_type='partial', roi_contour=roi,
                                                                          roi_hierarchy=roi_hierarchy,
                                                                          object_contour=contours,
                                                                          obj_hierarchy=contour_hierarchy,
                                                                          device=device, debug=args.debug)

    # Analyze only images with plants present
    if len(roi_objects) > 0:
        # Object combine kept objects
        device, plant_contour, plant_mask = pcv.object_composition(img=img, contours=roi_objects, hierarchy=hierarchy,
                                                                   device=device, debug=args.debug)

        if args.writeimg:
            # Save the plant mask if requested
            pcv.print_image(img=plant_mask, filename=outfile + "_mask.png")

        # Find shape properties, output shape image
        device, shape_header, shape_data, shape_img = pcv.analyze_object(img=img, imgname=args.image, obj=plant_contour,
                                                                         mask=plant_mask, device=device,
                                                                         debug=args.debug,
                                                                         filename=outfile)
        # Set the boundary line based on the camera zoom level
        if "z1500" in args.image:
            line_position = 930
        elif "z2500" in args.image:
            line_position = 885
        else:
            pcv.fatal_error("Image {0} has an unsupported zoom level.".format(args.image))

        # Shape properties relative to user boundary line
        device, boundary_header, boundary_data, boundary_img = pcv.analyze_bound_horizontal(img=img, obj=plant_contour,
                                                                                            mask=plant_mask,
                                                                                            line_position=line_position,
                                                                                            device=device,
                                                                                            debug=args.debug,
                                                                                            filename=outfile)

        # Determine color properties: Histograms, Color Slices and Pseudocolored Images,
        # output color analyzed images
        device, color_header, color_data, color_img = pcv.analyze_color(img=img, imgname=args.image, mask=plant_mask,
                                                                        bins=256,
                                                                        device=device, debug=args.debug,
                                                                        hist_plot_type=None,
                                                                        pseudo_channel="v", pseudo_bkg="img",
                                                                        resolution=300,
                                                                        filename=outfile)

        # Output shape and color data
        result = open(args.result, "a")
        result.write('\t'.join(map(str, shape_header)) + "\n")
        result.write('\t'.join(map(str, shape_data)) + "\n")
        for row in shape_img:
            result.write('\t'.join(map(str, row)) + "\n")
        result.write('\t'.join(map(str, color_header)) + "\n")
        result.write('\t'.join(map(str, color_data)) + "\n")
        result.write('\t'.join(map(str, boundary_header)) + "\n")
        result.write('\t'.join(map(str, boundary_data)) + "\n")
        result.write('\t'.join(map(str, boundary_img)) + "\n")
        for row in color_img:
            result.write('\t'.join(map(str, row)) + "\n")
        result.close()


if __name__ == '__main__':
    main()
