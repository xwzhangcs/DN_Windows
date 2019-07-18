#######################################################################
# Generate HTML file that shows the input and output images to compare.

from os.path import isfile, join
from PIL import Image
import os
import argparse
import numpy as np
import json
import subprocess
import sys
import shutil
import glob
import pandas as pd
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
from skimage import io


def main(facades_dir, chips_dir, segs_dir, dilates_dir, algins_dir, dnnsIn_dir, dnnsOut_dir, html_file_name):

	# Create the html file
	html_file = "<html>\n"
	html_file += "  <head>\n"
	html_file += "    <style>\n"
	html_file += "    table {\n"
	html_file += "      color: #333;\n"
	html_file += "      font-family: Helvetica, Arial, sans-serif;\n"
	html_file += "      border-collapse: collapse;\n"
	html_file += "      border-spacing: 0;\n"
	html_file += "    }\n"
	html_file += "    \n"
	html_file += "    td, th {\n"
	html_file += "      border: 1px solid #CCC;\n"
	html_file += "      padding: 5px;\n"
	html_file += "    }\n"
	html_file += "    \n"
	html_file += "    th {\n"
	html_file += "      background: #F3F3F3;\n"
	html_file += "      font-weight: bold;\n"
	html_file += "    }\n"
	html_file += "    \n"
	html_file += "    td {\n"
	html_file += "      text-align: center;\n"
	html_file += "    }\n"
	html_file += "    \n"
	html_file += "    tr:hover {\n"
	html_file += "      background-color: #eef;\n"
	html_file += "      border: 3px solid #00f;\n"
	html_file += "      font-weight: bold;\n"
	html_file += "    }\n"
	html_file += "    </style>\n"
	html_file += "  </head>\n"
	html_file += "<body>\n"
	html_file += "  <table>\n"
	html_file += "    <tr>\n"
	html_file += "      <th>Image.</th>\n"
	html_file += "      <th>Real Facade.</th>\n"
	# html_file += "      <th>Facade histeq.</th>\n"
	html_file += "      <th>Chip.</th>\n"
	# html_file += "      <th>Chip histeq.</th>\n"
	html_file += "      <th>Segmented.</th>\n"
	html_file += "      <th>Dilate-processed.</th>\n"
	html_file += "      <th>Align-processed.</th>\n"
	html_file += "      <th>DNN In.</th>\n"
	html_file += "      <th>DNN.</th>\n"

	facades = sorted(os.listdir(facades_dir))
	#df_score = pd.read_csv(score_file)
	#df_conf = pd.read_csv(confidence_file)
	for i in range(len(facades)):
		facade_file = facades_dir + '/' + facades[i]
		# facadehist_file = facadehist_dir + '/' + facades[i]
		chip_file = chips_dir + '/' + facades[i]
		# chiphist_file = chiphist_dir + '/' + facades[i]
		seg_file = segs_dir + '/' + facades[i]
		dilate_file = dilates_dir + '/' + facades[i]
		align_file = algins_dir + '/' + facades[i]
		dnnIn_file = dnnsIn_dir + '/' + facades[i]
		dnnOut_file = dnnsOut_dir + '/' + facades[i]
		html_file += "    <tr>\n"
		html_file += "      <td>" + facades[i] + "</td>\n"
		html_file += "      <td><a href=\"" + facade_file + "\"><img src=\"" + facade_file + "\"/></a></td>\n"
		# html_file += "      <td><a href=\"" + facadehist_file + "\"><img src=\"" + facadehist_file + "\"/></a></td>\n"
		# html_file += "      <td>" + str(df_score.loc[df_score['image'] == facades[i]].iloc[0, 1]) + "</td>\n"
		html_file += "      <td><a href=\"" + chip_file + "\"><img src=\"" + chip_file + "\"/></a></td>\n"
		# html_file += "      <td><a href=\"" + chiphist_file + "\"><img src=\"" + chiphist_file + "\"/></a></td>\n"
		html_file += "      <td><a href=\"" + seg_file + "\"><img src=\"" + seg_file + "\"/></a></td>\n"
		html_file += "      <td><a href=\"" + dilate_file + "\"><img src=\"" + dilate_file + "\"/></a></td>\n"
		html_file += "      <td><a href=\"" + align_file + "\"><img src=\"" + align_file + "\"/></a></td>\n"
		html_file += "      <td><a href=\"" + dnnIn_file + "\"><img src=\"" + dnnIn_file + "\"/></a></td>\n"
		html_file += "      <td><a href=\"" + dnnOut_file + "\"><img src=\"" + dnnOut_file + "\"/></a></td>\n"
		html_file += "    </tr>\n"

	html_file += "  </table>\n"
	html_file += "</body>\n"
	html_file += "</html>\n"
		
	# Save the html file
	with open(html_file_name, "w") as output_file:
		output_file.write(html_file)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("facades_dir", help="path to input image folder (e.g., input_data)")
	# parser.add_argument("facadehist_dir", help="path to input image folder (e.g., input_data)")
	# parser.add_argument("score_file", help="path to input image folder (e.g., input_data)")
	parser.add_argument("chips_dir", help="path to input image folder (e.g., input_data)")
	# parser.add_argument("chiphist_dir", help="path to input image folder (e.g., input_data)")
	# parser.add_argument("confidence_file", help="path to input image folder (e.g., input_data)")
	parser.add_argument("segs_dir", help="path to input image folder (e.g., input_data)")
	parser.add_argument("dilates_dir", help="path to input image folder (e.g., input_data)")
	parser.add_argument("algins_dir", help="path to input image folder (e.g., input_data)")
	parser.add_argument("dnnsIn_dir", help="path to input image folder (e.g., input_data)")
	parser.add_argument("dnnsOut_dir", help="path to input image folder (e.g., input_data)")
	# parser.add_argument("dnnshist_dir", help="path to input image folder (e.g., input_data)")
	parser.add_argument("html_file_name", help="path to output html filename")
	args = parser.parse_args()

	main(args.facades_dir, args.chips_dir, args.segs_dir, args.dilates_dir, args.algins_dir, args.dnnsIn_dir, args.dnnsOut_dir, args.html_file_name)
