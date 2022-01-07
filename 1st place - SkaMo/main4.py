import json
import math
import os

import numpy as np
import rasterio
from skimage.measure import CircleModel, perimeter
from skimage.segmentation import find_boundaries, flood

# Directory to store working files
work_dir = "../features/"
os.makedirs ( work_dir, exist_ok = True )

# Train and test data path
train_path = "ref_south_africa_crops_competition_v1_train_labels"
test_path = "ref_south_africa_crops_competition_v1_test_labels"
tt_paths = [ train_path, test_path ]

# Dictionary of all fields, it maps int ( field_id ) -> Field
field_dict_fname = work_dir + "fields.csv"
field_dict = { }

# Dictionaries that maps tile id to tiles for training and testing tile sets
train_tiles_fname = work_dir + "train_tiles.csv"
test_tiles_fname = work_dir + "test_tiles.csv"
train_tiles = { }
test_tiles = { }

# Labels dictionary
labels_dict_fname = "ref_south_africa_crops_competition_v1_train_labels/_common/raster_values.json"
label_num_dict = { }
num_label_dict = { }

# Overlap - store pixel offsets for overlapped tiles
overlap_tiles_fname = work_dir + "overlap_tiles.csv"

# Overlap - store pixel counts for overlapped fields
overlap_fields_fname = work_dir + "overlap_fields.csv"

# Overlap - based on overlap determine initial probability as a feature
overlap_features_fname = work_dir + "overlap_features.csv"

# Training set labels
train_labels_fname = "ref_south_africa_crops_competition_v1_train_labels/_common/field_info_train.csv"

# Field neighbours
field_neighbours_fname = work_dir + "neighbours.csv"

# Number of fields and maximum field id
n_fields = 122408
max_field_id = 122736

# Field lookup array
# First column is row in feature matrix, so field_lookup [ field_id, 0 ] = row in feature matrix
# Second column is field id of row in feature matrix, so field_lookup[ 5, 1 ] = field_id of field stored in row 5
field_lookup = np.zeros ( [ max_field_id + 1, 2 ], dtype = int )
field_lookup.fill ( -1 )

# Bands we will calculate for
band_names = [ "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A", "EVI", "MI", "NDVI", "SAVI", "VH", "VV" ]
band_name_index = { }
n_bands = len ( band_names )

# s2 bands only
s2_band_names = [ "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A", "EVI", "MI", "NDVI", "SAVI" ]
s2_band_name_index = { }
n_s2_bands = len ( s2_band_names )

# Dates we will calculate for
date_names = [ "4a", "4b", "5a", "5b", "6a", "6b", "7a", "7b", "8a", "8b", "9a", "9b", "10a", "10b", "11a", "11b" ]
date_name_index = { }
n_dates = len ( date_names )

# Support matrix we will use to calculate features later
# Entries are number of obs, min, max, sum and sum of squares
data_support = np.zeros ( [ n_fields, n_dates, n_bands, 5 ], dtype = float )
data_support_fname = work_dir + "data_support.npy"

# Support for s2 only
s2_support = np.zeros ( [ n_fields, n_dates, n_s2_bands, 5 ], dtype = float )
s2_support_fname = work_dir + "s2_support.npy"

# Tolerance when looking for tile neighbours
tile_tol_dx = 0.001
tile_tol_dy = 0.001

# Arbitrary scaling to apply when calculating field size
scale_size = 20000

# Radius to use when adding neighbouring fields
neighbour_radius = 0.05

# Arbitrary scaling to apply to position
scale_pos_x_add = -17.8
scale_pos_y_add = 34.2
scale_pos_x_mul = 1
scale_pos_y_mul = 1

# Tile
class Tile :
	in_train = None
	bbox = None

	# Files
	tile_file = None
	field_id_file = None

	# Overlapped tiles, taken from train and test sets
	# Also store overlap tile's pixel offsets
	right_train_id = None
	right_train_overlap = None
	right_test_id = None
	right_test_overlap = None
	bottom_train_id = None
	bottom_train_overlap = None
	bottom_test_id = None
	bottom_test_overlap = None

# Field
class Field :
	in_train = None
	row = None
	tile_id = None
	field_pixels = None
	field_size = None
	field_center = None
	label = 0
	fragments = None
	perimeter = None
	perimeter_to_size = None
	circular_measure = None

	# Overlap

	# The number of pixels this field has in the border
	border_pixels = 0

	# The count of the total number of pixels that overlap
	overlap_count = 0

	# None if no overlap, otherwise a dictionary, with sum of all values = overlap_count
	# The dictionary has as key the overlapping field's id and as value the number of pixels that overlap
	overlap = None

	# Neighbours
	neighbour_ids = None
	neighbour_distances = None

# Utility to open, load, close and return a json file
def load_json_file ( fname ) :
	file = open ( fname )
	x = json.load ( file )
	file.close ()
	return x

# Utility to open, load, close and return a raster
def load_raster ( fname ) :
	file = rasterio.open ( fname )
	x = file.read ( 1 )
	file.close ()
	return x

# Utility to return a pixel value
def img_pixel_value ( img, x, y ) :
	if x >= img.shape [ 1 ] or y >= img.shape [ 0 ] or x < 0 or y < 0 :
		return -1
	return int ( img [ y, x ] )

# Update the dictionary that maps field ids to fields
def update_field_dict () :
	# Load the fields and tiles
	if os.path.isfile ( field_dict_fname ) :

		# Read from file
		print ( "Reading fields" )
		fdin = open ( field_dict_fname )

		# Skip header
		fdin.readline ()

		row = 0

		for line in fdin :

			fields = line.strip ().split ( "," )

			field_id = int ( fields [ 0 ] )

			field = Field ()

			field.row = row
			field.in_train = fields [ 1 ] == 1
			field.tile_id = int ( fields [ 2 ] )
			field.field_pixels = int ( fields [ 3 ] )
			field.field_size = float ( fields [ 4 ] )
			field.field_center = [ float ( fields [ 5 ] ), float ( fields [ 6 ] ) ]
			field.fragments = float ( fields [ 7 ] )
			field.perimeter = float ( fields [ 8 ] )
			field.perimeter_to_size = float ( fields [ 9 ] )
			field.circular_measure = float ( fields [ 10 ] )

			field_dict [ field_id ] = field
			row = row + 1

		fdin.close ()

		# Read tiles
		print ( "Reading tiles" )
		in_train = True

		for tile_fname in [ train_tiles_fname, test_tiles_fname ] :

			tdin = open ( tile_fname )

			# Skip header
			tdin.readline ()

			for line in tdin :

				fields = line.strip ().split ( "," )

				tile_id = int ( fields [ 0 ] )

				tile = Tile ()
				tile.in_train = in_train
				tile.bbox = [ float ( fields [ 1 ] ), float ( fields [ 2 ] ), float ( fields [ 3 ] ), float ( fields [ 4 ] ) ]
				tile.tile_file = fields [ 5 ]
				tile.field_id_file = fields [ 6 ]

				if in_train :

					train_tiles [ tile_id ] = tile

				else :

					test_tiles [ tile_id ] = tile

			# Done
			tdin.close ()

			# Move to test
			in_train = False

	else :

		# First load for training set, then testing set
		print ( "Building dictionaries" )
		in_train = True
		tt = 0
		tn = 0

		for path in tt_paths :

			if in_train :
				print ( "\tTiles in training set" )
			else :
				print ( "\tTiles in test set" )

			labels_collection = load_json_file ( f"{path}/collection.json" )

			for link in labels_collection [ "links" ] :

				if link [ "rel" ] == "item" :

					item_link = link [ "href" ]
					item_path = f"{path}/{item_link}"
					current_path = os.path.dirname ( item_path )
					item = load_json_file ( item_path )
					tile_id = int ( item [ "id" ].split ( "_" ) [ -1 ] )

					# Create a tile
					tile = Tile ()

					tile.tile_file = item_path
					tile.in_train = in_train
					tile.bbox = item [ "bbox" ]

					# Update tile coordinates
					tile_width = tile.bbox [ 2 ] - tile.bbox [ 0 ]
					tile_height = tile.bbox [ 3 ] - tile.bbox [ 1 ]

					if in_train :

						train_tiles [ tile_id ] = tile

					else :

						test_tiles [ tile_id ] = tile

					# Load field ids
					for asset_key, asset in item [ "assets" ].items () :

						if asset_key == "field_ids" :

							asset_link = asset [ "href" ]
							asset_path = f"{current_path}/{asset_link}"
							tile.field_id_file = asset_path
							field_ids = load_raster ( asset_path )
							field_height, field_width = field_ids.shape

							# Size of a single pixel
							pixel_width = tile_width / field_width
							pixel_height = tile_height / field_height
							pixel_size = scale_size * pixel_width * pixel_height

							# Create a grid that can be used to calculate position
							pos_xx = tile.bbox [ 0 ] + 0.5 * pixel_width + pixel_width * np.arange ( 0, field_width )
							pos_yy = tile.bbox [ 3 ] - 0.5 * pixel_height + pixel_height * np.arange ( 0, field_height )
							pos_xx, pos_yy = np.meshgrid ( pos_xx, pos_yy )

							# Field pixel count, used to determine field size
							field_counts = { }

							for field_id in field_ids.flatten () :

								i = int ( field_id )

								if i > 0 :

									n = field_counts.get ( i, 0 )

									if n > 0 :

										field_counts [ i ] = n + 1

									else :

										field_counts [ i ] = 1

							# Now we can calculate the size and position of each field
							for field_id, count in field_counts.items () :

								field = Field ()

								field.in_train = in_train
								field.tile_id = tile_id
								field.field_pixels = count
								field.field_size = count * pixel_size
								field_mask = field_ids == field_id
								field.field_center = [ (pos_xx [ field_mask ].mean () + scale_pos_x_add) * scale_pos_x_mul,
										       (pos_yy [ field_mask ].mean () + scale_pos_y_add) * scale_pos_y_mul ]

								# Some image processing
								o, b, i, p = outside_boundary_inside_perimeter ( field_mask )
								s = np.count_nonzero ( i )
								field.fragments = 1 - count / s
								field.perimeter = p
								field.perimeter_to_size = p / s
								field.circular_measure = fit_circle ( b )

								field_dict [ field_id ] = field

			# Move to test
			in_train = False

		# Store
		print ( "Writing tiles" )

		in_train = True
		for fname in [ train_tiles_fname, test_tiles_fname ] :

			fdout = open ( fname, "w" )

			# Header
			fdout.write ( "tile_id,bbox0,bbox1,bbox2,bbox3,tile path,field ids path\n" )

			if in_train :

				dict = train_tiles

			else :

				dict = test_tiles

			for tile_id in sorted ( dict.keys () ) :

				tile = dict [ tile_id ]

				fdout.write ( f"{tile_id},{tile.bbox [ 0 ]},{tile.bbox [ 1 ]},{tile.bbox [ 2 ]},{tile.bbox [ 3 ]}" )
				fdout.write ( "," )
				fdout.write ( tile.tile_file )
				fdout.write ( "," )
				fdout.write ( tile.field_id_file )

				fdout.write ( "\n" )

			# Done
			fdout.close ()

			# Move to test
			in_train = False

		# Fields
		print ( "Writing fields" )
		fdout = open ( field_dict_fname, "w" )

		# Header
		fdout.write ( "field_id,train,tile_id,pixel size,field size,center x,center y,fragmentation,perimeter,perimeter to size,circular measure\n" )

		row = 0
		for field_id in sorted ( field_dict.keys () ) :
			field = field_dict [ field_id ]
			field.row = row

			if field.in_train :

				in_train = 1

			else :

				in_train = 0

			fdout.write ( f"{field_id},{in_train},{field.tile_id},{field.field_pixels},{field.field_size},{field.field_center [ 0 ]},{field.field_center [ 1 ]},{field.fragments},{field.perimeter},{field.perimeter_to_size},{field.circular_measure}\n" )
			row = row + 1

		fdout.close ()

	# Load the labels
	dict = load_json_file ( labels_dict_fname )

	for key, value in dict.items () :

		label_num_dict [ value ] = int ( key )
		num_label_dict [ int ( key ) ] = value

	# Load and assign training labels
	fin = open ( train_labels_fname )

	# Skip header
	fin.readline ()

	for line in fin :

		fields = line.strip ().split ( "," )
		field_dict [ int ( fields [ 0 ] ) ].label = label_num_dict [ fields [ 1 ] ]

	fin.close ()

# Given two bboxes, return the intersection
def bbox_intersect ( bbox0, bbox1 ) :
	x0 = max ( bbox0 [ 0 ], bbox1 [ 0 ] )
	x1 = min ( bbox0 [ 2 ], bbox1 [ 2 ] )

	y0 = max ( bbox0 [ 1 ], bbox1 [ 1 ] )
	y1 = min ( bbox0 [ 3 ], bbox1 [ 3 ] )

	if x0 <= x1 and y0 <= y1 :

		return [ x0, y0, x1, y1 ]

	return None

# Return the area of a bbox
def bbox_area ( bbox ) :
	if bbox :

		w = bbox [ 2 ] - bbox [ 0 ]
		h = bbox [ 3 ] - bbox [ 1 ]

		return w * h

	else :

		return 0

# Load all tile coordinates for a given label
def load_tile_coordinates ( label_id, coord_dict ) :
	labels_collection = load_json_file ( f"{label_id}/collection.json" )

	for link in labels_collection [ "links" ] :

		if link [ "rel" ] == "item" :

			item_link = link [ "href" ]
			item_path = f"{label_id}/{item_link}"
			item = load_json_file ( item_path )
			tile_id = int ( item [ "id" ].split ( "_" ) [ -1 ] )
			bbox = item [ "bbox" ]

			coord_dict [ tile_id ] = bbox

# Find tile that is next to and to the right of a given tile
def tile_to_right ( left_bbox, tile_dict ) :
	max_score = 0
	max_id = None

	for right_id, right_tile in tile_dict.items () :

		dx = abs ( left_bbox [ 2 ] - right_tile.bbox [ 0 ] )
		dy = abs ( left_bbox [ 1 ] - right_tile.bbox [ 1 ] )

		if dx < tile_tol_dx and dy < tile_tol_dy :

			score = bbox_area ( bbox_intersect ( left_bbox, right_tile.bbox ) )

			if score > max_score :
				if max_score > 0 :
					print ( "Found another right" )
				max_score = score
				max_id = right_id

	return max_id

# Find tile that is next to and below a given tile
def tile_to_bottom ( top_bbox, tile_dict ) :
	max_score = 0
	max_id = None

	for bottom_id, bottom_tile in tile_dict.items () :

		dx = abs ( top_bbox [ 0 ] - bottom_tile.bbox [ 0 ] )
		dy = abs ( top_bbox [ 1 ] - bottom_tile.bbox [ 3 ] )

		if dx < tile_tol_dx and dy < tile_tol_dy :

			score = bbox_area ( bbox_intersect ( top_bbox, bottom_tile.bbox ) )
			if score > max_score :
				if max_score > 0 :
					print ( "Found another bottom" )
				max_score = score
				max_id = bottom_id

	return max_id

# Return a measure of the overlap between two images
def measure_overlap ( img0, img1, x0, y0, x1, y1, w, h ) :
	return np.count_nonzero ( img0 [ y0 :y0 + h, x0 :x0 + w ] * img1 [ y1 :y1 + h, x1 :x1 + w ] )

# Find optimal overlap on the right by varying both x and y
def opt_right_overlapxy ( left, right, width, height ) :
	full_height = left.shape [ 0 ]
	full_width = left.shape [ 1 ]

	tolx = 1 + width // 2
	toly = full_height - height

	x00 = full_width - width

	maxm = 0
	maxx0 = 0
	maxy0 = 0
	maxx1 = 0
	maxy1 = 0

	for x0 in range ( tolx - 1 ) :
		for x1 in range ( tolx - 1 ) :
			for y0 in range ( toly ) :
				for y1 in range ( toly ) :

					m = measure_overlap ( left, right, x00 + x0, y0, x1, y1, tolx, height )

					if m > maxm :
						maxm = m
						maxx0 = x00 + x0
						maxy0 = y0
						maxx1 = x1
						maxy1 = y1

	return maxm, maxx0, maxy0, maxx1, maxy1, tolx, height

# Find optimal overlap on the bottom along both x and y
def opt_bottom_overlapxy ( top, bottom, width, height ) :
	full_height = top.shape [ 0 ]
	full_width = top.shape [ 1 ]

	tolx = full_width - width
	toly = 1 + height // 2

	y00 = full_height - height

	maxm = 0
	maxx0 = 0
	maxy0 = 0
	maxx1 = 0
	maxy1 = 0

	for x0 in range ( tolx ) :
		for x1 in range ( tolx ) :
			for y0 in range ( toly - 1 ) :
				for y1 in range ( toly - 1 ) :
					m = measure_overlap ( top, bottom, x0, y00 + y0, x1, y1, width, toly )

					if m > maxm :
						maxm = m
						maxx0 = x0
						maxy0 = y00 + y0
						maxx1 = x1
						maxy1 = y1

	return maxm, maxx0, maxy0, maxx1, maxy1, width, toly

# Find optimal overlap along both dimensions
def opt_overlapxy ( tile0, tile1, width, height ) :
	if width > height :

		m, x0, y0, x1, y1, w, h = opt_bottom_overlapxy ( tile0, tile1, width, height )

	else :

		m, x0, y0, x1, y1, w, h = opt_right_overlapxy ( tile0, tile1, width, height )

	return x0, y0, x1, y1, w, h

# Update the overlap between two tiles
def update_overlap_tiles ( right, tile0, raster0, tile1 ) :
	# See if we have any overlap
	overlap = bbox_intersect ( tile0.bbox, tile1.bbox )

	if overlap :

		raster1 = load_raster ( tile1.field_id_file )

		# Overlap width and height
		# Subtract for a bit of tolerance
		w = int ( raster0.shape [ 1 ] * (overlap [ 2 ] - overlap [ 0 ]) / (tile0.bbox [ 2 ] - tile0.bbox [ 0 ]) + 0.5 ) - 1
		h = int ( raster0.shape [ 0 ] * (overlap [ 3 ] - overlap [ 1 ]) / (tile0.bbox [ 3 ] - tile0.bbox [ 1 ]) + 0.5 ) - 1

		# Find optimal overlap
		x0, y0, x1, y1, w0, h0 = opt_overlapxy ( raster0, raster1, w, h )

		# Find overlap and border counts
		border_pixel_counts = { }
		overlap_proposals = { }

		for x in range ( w0 ) :

			for y in range ( h0 ) :

				id0 = img_pixel_value ( raster0, x0 + x, y0 + y )
				id1 = img_pixel_value ( raster1, x1 + x, y1 + y )

				if id0 > 0 :

					n = border_pixel_counts.get ( id0, 0 )
					border_pixel_counts [ id0 ] = n + 1

				if id1 > 0 :

					n = border_pixel_counts.get ( id1, 0 )
					border_pixel_counts [ id1 ] = n + 1

				if id0 > 0 and id1 > 0 and id0 != id1 :

					# Get current proposal overlap count
					overlap_counts = overlap_proposals.get ( id0, None )

					# Update overlap count
					if overlap_counts is None :

						overlap_counts = { }
						overlap_counts [ id1 ] = 1
						overlap_proposals [ id0 ] = overlap_counts

					else :

						n = overlap_counts.get ( id1, 0 )
						overlap_counts [ id1 ] = n + 1

		# Update border pixel counds
		for field_id, pixel_count in border_pixel_counts.items () :

			field_dict [ field_id ].border_pixels = field_dict [ field_id ].border_pixels + pixel_count

		# Resolve the overlap proposals
		for field_id, overlap_counts in overlap_proposals.items () :

			field = field_dict [ field_id ]

			if field.overlap_count == 0 :

				field.overlap = { }

			for overlap_id, overlap_count in overlap_counts.items () :

				overlap = field_dict [ overlap_id ]

				if overlap.overlap_count == 0 :

					overlap.overlap = { }

				# Update field
				field.overlap_count = field.overlap_count + overlap_count
				n = field.overlap.get ( overlap_id, 0 )
				field.overlap [ overlap_id ] = n + overlap_count

				# Update overlap field
				overlap.overlap_count = overlap.overlap_count + overlap_count
				n = overlap.overlap.get ( field_id, 0 )
				overlap.overlap [ field_id ] = n + overlap_count

		# Return overlap
		return [ x0, y0, x1, y1, w0, h0 ]

	# No overlap
	return None

# Update the overlap fields
def update_overlap_fields () :
	# Load the overlap results if available
	if os.path.isfile ( overlap_tiles_fname ) :

		print ( "Reading overlapped tiles" )

		# Load tile overlap offsets
		oin = open ( overlap_tiles_fname )

		# Header
		oin.readline ()

		# Loop
		for line in oin :
			fields = line.strip ().split ( "," )

			in_train = int ( fields [ 0 ] )
			tile_id = int ( fields [ 1 ] )
			right = int ( fields [ 2 ] )
			train = int ( fields [ 3 ] )
			overlap_id = int ( fields [ 4 ] )

			x0 = int ( fields [ 5 ] )
			y0 = int ( fields [ 6 ] )
			x1 = int ( fields [ 7 ] )
			y1 = int ( fields [ 8 ] )
			w = int ( fields [ 9 ] )
			h = int ( fields [ 10 ] )

			overlap = [ x0, y0, x1, y1, w, h ]

			if in_train > 0 :

				tile = train_tiles [ tile_id ]

			else :

				tile = test_tiles [ tile_id ]

			if right > 0 :
				if train > 0 :
					tile.right_train_id = overlap_id
					tile.right_train_overlap = overlap
				else :
					tile.right_test_id = overlap_id
					tile.right_test_overlap = overlap
			else :
				if train > 0 :
					tile.bottom_train_id = overlap_id
					tile.bottom_train_overlap = overlap
				else :
					tile.bottom_test_id = overlap_id
					tile.bottom_test_overlap = overlap

		oin.close ()

		# Read from file
		print ( "Reading overlapped fields" )
		fin = open ( overlap_fields_fname )

		# Header
		fin.readline ()

		# Loop
		for line in fin :
			fields = line.strip ().split ( "," )
			field_id = int ( fields [ 0 ] )
			pixels = int ( fields [ 1 ] )
			overlap_id = int ( fields [ 2 ] )
			overlap_count = int ( fields [ 3 ] )

			field = field_dict [ field_id ]
			field.boder_pixels = pixels

			if overlap_id > 0 and overlap_count > 0 :

				if field.overlap_count == 0 :

					field.overlap_count = overlap_count
					field.overlap = { }

				else :

					field.overlap_count = field.overlap_count + overlap_count

				field.overlap [ overlap_id ] = overlap_count

		fin.close ()

	else :

		print ( "Updating overlapped fields" )

		for tiles in [ train_tiles, test_tiles ] :

			for tile_id, tile in tiles.items () :

				raster = load_raster ( tile.field_id_file )

				# Overlap on right from train
				overlap_id = tile_to_right ( tile.bbox, train_tiles )

				if overlap_id is not None :

					overlap = update_overlap_tiles ( True, tile, raster, train_tiles [ overlap_id ] )

					if overlap is not None :

						tile.right_train_id = overlap_id
						tile.right_train_overlap = overlap

				# Overlap on right from test
				overlap_id = tile_to_right ( tile.bbox, test_tiles )

				if overlap_id is not None :

					overlap = update_overlap_tiles ( True, tile, raster, test_tiles [ overlap_id ] )

					if overlap is not None :

						tile.right_test_id = overlap_id
						tile.right_test_overlap = overlap

				# Overlap on bottom from train
				overlap_id = tile_to_bottom ( tile.bbox, train_tiles )

				if overlap_id is not None :

					overlap = update_overlap_tiles ( False, tile, raster, train_tiles [ overlap_id ] )

					if overlap is not None :

						tile.bottom_train_id = overlap_id
						tile.bottom_train_overlap = overlap

				# Overlap on bottom from test
				overlap_id = tile_to_bottom ( tile.bbox, test_tiles )

				if overlap_id is not None :

					overlap = update_overlap_tiles ( False, tile, raster, test_tiles [ overlap_id ] )

					if overlap is not None :

						tile.bottom_test_id = overlap_id
						tile.bottom_test_overlap = overlap

		# Now we can store
		fout = open ( overlap_tiles_fname, "w" )
		fout.write ( "in train,tile_id,right,train,overlap_id,x0,y0,x1,y1,w,h\n" )

		in_train = 1
		for dict in [ train_tiles, test_tiles ] :

			for tile_id, tile in dict.items () :

				if tile.right_train_id is not None :
					fout.write ( f"{in_train},{tile_id},1,1,{tile.right_train_id}" )
					for i in tile.right_train_overlap :
						fout.write ( "," )
						fout.write ( str ( i ) )
					fout.write ( "\n" )

				if tile.right_test_id is not None :
					fout.write ( f"{in_train},{tile_id},1,0,{tile.right_test_id}" )
					for i in tile.right_test_overlap :
						fout.write ( "," )
						fout.write ( str ( i ) )
					fout.write ( "\n" )

				if tile.bottom_train_id is not None :
					fout.write ( f"{in_train},{tile_id},0,1,{tile.bottom_train_id}" )
					for i in tile.bottom_train_overlap :
						fout.write ( "," )
						fout.write ( str ( i ) )
					fout.write ( "\n" )

				if tile.bottom_test_id is not None :
					fout.write ( f"{in_train},{tile_id},0,0,{tile.bottom_test_id}" )
					for i in tile.bottom_test_overlap :
						fout.write ( "," )
						fout.write ( str ( i ) )
					fout.write ( "\n" )

			in_train = 0

		fout.close ()

		# Overlap fields
		fout = open ( overlap_fields_fname, "w" )
		fout.write ( "field_id,border pixels,overlap_id,overlap_count\n" )

		for field_id, field in field_dict.items () :

			if field.overlap_count > 0 :

				for overlap_id, overlap_count in field.overlap.items () :

					fout.write ( f"{field_id},{field.border_pixels},{overlap_id},{overlap_count}\n" )

			elif field.border_pixels > 0 :

				fout.write ( f"{field_id},{field.border_pixels},0,0\n" )

		fout.close ()

# Given an image, fill the holes in it
# and return the outside mask, boundary
# and inside mask and estimated
# perimeter length
# Note the boundary is 2 pixels wide
def outside_boundary_inside_perimeter ( img ) :
	x0 = 0
	y0 = 0

	if img [ y0, x0 ] :

		x0 = img.shape [ 1 ] - 1
		y0 = img.shape [ 1 ] - 1

	outside = flood ( img, (y0, x0) )
	inside = ~outside
	boundary = find_boundaries ( inside )

	return outside, boundary, inside, perimeter ( inside )

# Update the overlap features
def update_overlap_features () :
	# Load the overlap results if available
	if not os.path.isfile ( overlap_features_fname ) :

		print ( "Creating overlapped features" )

		fout = open ( overlap_features_fname, "w" )
		fout.write ( "field_id,p1,p2,p3,p4,p5,p6,p7,p8,p9\n" )

		for field_id, field in field_dict.items () :

			if field.label == 0 :

				if field.overlap_count > 0 :

					total_count = 0
					label_counts = [ 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

					for overlap_id, overlap_count in field.overlap.items () :

						overlap = field_dict [ overlap_id ]

						if overlap.label > 0 :

							total_count = total_count + overlap_count
							label_counts [ overlap.label - 1 ] = label_counts [ overlap.label - 1 ] + overlap_count

					if total_count > 0 :

						fout.write ( str ( field_id ) )

						for label_count in label_counts :

							fout.write ( "," )
							fout.write ( str ( label_count / total_count ) )

						fout.write ( "\n" )

		fout.close ()

# Given the border of an image, return the goodness of fit of a circle
def fit_circle ( boundary ) :
	total = np.count_nonzero ( boundary )

	if total > 10 :

		xy = np.zeros ( [ total, 2 ], dtype = int )
		i = 0

		for x in range ( boundary.shape [ 1 ] ) :
			for y in range ( boundary.shape [ 0 ] ) :
				if boundary [ y, x ] :
					xy [ i, 0 ] = x
					xy [ i, 1 ] = y
					i = i + 1

		# Fit circle model
		circle = CircleModel ()

		if circle.estimate ( xy ) :

			return 1 / (1 + circle.residuals ( xy ).std ())

	return 0.0

# Update the field lookup array
def update_field_lookup () :
	row = 0

	for field_id in sorted ( field_dict.keys () ) :

		field_lookup [ row, 0 ] = field_id
		field_lookup [ field_id, 1 ] = row
		row = row + 1

# Update name indices
def update_name_indices () :
	for i, date_name in enumerate ( date_names ) :
		date_name_index [ date_name ] = i

	for i, band_name in enumerate ( band_names ) :
		band_name_index [ band_name ] = i

	for i, s2_band_name in enumerate ( s2_band_names ) :
		s2_band_name_index [ s2_band_name ] = i

# Given a date, return its name
def date_name ( date ) :
	month = int ( date [ 5 :7 ] )
	day = int ( date [ 8 :10 ] )

	if day <= 15 :

		return str ( month ) + 'a'

	else :

		return str ( month ) + 'b'

# Given a tile, update the data support with observations from its images
def calc_tile ( tile ) :
	current_path = os.path.dirname ( tile.tile_file )
	tile_assets = load_json_file ( tile.tile_file )
	field_ids = load_raster ( tile.field_id_file ).flatten ()

	for link in tile_assets [ "links" ] :

		if link [ "rel" ] == "source" :
			inner_link = link [ "href" ]
			inner_path = current_path + "/" + inner_link
			asset_dir = os.path.dirname ( inner_path )

			inner_item = load_json_file ( inner_path )

			# Get the date and index
			datetime = inner_item [ "properties" ] [ "datetime" ]
			dname = date_name ( datetime )
			dindex = date_name_index [ dname ]

			# Observations for this date
			obs = { }

			# Read observations from assets
			for asset_key in inner_item [ "assets" ] :

				asset_value = inner_item [ "assets" ] [ asset_key ]
				asset_link = asset_value [ "href" ]
				asset_path = asset_dir + "/" + asset_link
				obs [ asset_key ] = load_raster ( asset_path ).flatten ()

			# Apply cloud mask, if any
			clm = obs.get ( "CLM", None )
			if clm is None :

				# No cloud mask, this is s1
				mask = None

			else :

				# Cloud mask so this must be s2
				mask = clm == 0

				# Calculate composite bands
				n = clm.shape [ 0 ]
				ndvi = np.zeros ( n, dtype = float )
				evi = np.zeros ( n, dtype = float )
				savi = np.zeros ( n, dtype = float )
				mi = np.zeros ( n, dtype = float )

				b2 = obs [ "B02" ]
				b4 = obs [ "B04" ]
				b8 = obs [ "B08" ]
				b8a = obs [ "B8A" ]
				b11 = obs [ "B11" ]

				b48s = b4 + b8
				b48as = b4 + b8a
				b84d = b8 - b4

				evid = b8 + 6 * b4 - 7.5 * b2 + 1
				savid = (b48s + 0.725) * 1.725

				tm = mask & (b48s > 0)
				ndvi [ tm ] = b84d [ tm ] / b48s [ tm ]

				tm = mask & (evid != 0)
				evi [ tm ] = 2.5 * b84d [ tm ] / evid [ tm ]

				tm = mask & (savid != 0)
				savi [ tm ] = b84d [ tm ] / savid [ tm ]

				tm = mask & (b48as > 0)
				mi [ tm ] = (b8a [ tm ] - b11 [ tm ]) / b48as [ tm ]

				obs [ "NDVI" ] = ndvi
				obs [ "EVI" ] = evi
				obs [ "SAVI" ] = savi
				obs [ "MI" ] = mi

			# Now we can update for all the bands
			for field_id in np.unique ( field_ids ) :

				if field_id > 0 :

					row = field_lookup [ field_id, 1 ]

					for band_name in obs.keys () :

						bindex = band_name_index.get ( band_name, -1 )

						if bindex >= 0 :

							# Extract the observations for this field
							# Apply both the cloud mask and the field id mask
							if mask is None :

								x = obs [ band_name ] [ field_ids == field_id ]

							else :

								x = obs [ band_name ] [ (field_ids == field_id) & mask ]

							n = x.shape [ 0 ]

							if n > 0 :

								old_n = data_support [ row, dindex, bindex, 0 ]

								mn = x.min ()
								mx = x.max ()
								s1 = x.sum ()
								s2 = (x * x).sum ()

								# Update
								data_support [ row, dindex, bindex, 0 ] = old_n + n

								if old_n == 0 or mn < data_support [ row, dindex, bindex, 1 ] :

									data_support [ row, dindex, bindex, 1 ] = mn

								if old_n == 0 or mx > data_support [ row, dindex, bindex, 2 ] :

									data_support [ row, dindex, bindex, 2 ] = mx

								data_support [ row, dindex, bindex, 3 ] = data_support [ row, dindex, bindex, 3 ] + s1
								data_support [ row, dindex, bindex, 4 ] = data_support [ row, dindex, bindex, 4 ] + s2

								# Transfer to s2 if valid
								s2index = s2_band_name_index.get ( band_name, -1 )

								if s2index >= 0 :
									s2_support [ row, dindex, s2index, : ] = data_support [ row, dindex, bindex, : ]

# Update the data support
def update_data_support () :
	# Load the results if available
	if not os.path.isfile ( data_support_fname ) :

		# Need to recalculate
		print ( f"Calculting for train set" )
		for tile in train_tiles.values () :

			calc_tile ( tile )

		print ( f"Calculting for test set" )
		for tile in test_tiles.values () :

			calc_tile ( tile )

		np.save ( data_support_fname, data_support )
		np.save ( s2_support_fname, s2_support )

	else :

		print ( f"Loading data support" )

		# Load like this to retain global allocated array and to test dimensions
		data_support [ :, :, :, : ] = np.load ( data_support_fname )
		s2_support [ :, :, :, : ] = np.load ( s2_support_fname )

# Update the field neighbours dictionary
def update_field_neighbours () :
	# Load the results if available
	if not os.path.isfile ( field_neighbours_fname ) :

		# Need to recalculate
		print ( f"Calculating distance to neihbours" )

		# Write results as we calculate
		out = open ( field_neighbours_fname, "w" )
		out.write ( "field_id,neighbour_id,distance\n" )

		# Keep track of number of neighbours
		sum = 0
		sorted_field_ids = sorted ( field_dict.keys () )

		# Speed it up
		r2 = neighbour_radius * neighbour_radius
		tile_radius = 5 * neighbour_radius
		n2 = tile_radius * tile_radius

		for i, field_id in enumerate ( sorted_field_ids ) :

			if i % 10 == 0 :

				print ( f"\tNow at {i + 1}/{len ( sorted_field_ids )} - have noted {sum} neighbours thus far" )

			field = field_dict [ field_id ]

			field.neighbour_ids = [ ]
			field.neighbour_distances = [ ]

			fx = field.field_center [ 0 ]
			fy = field.field_center [ 1 ]

			# To speed it up, keep track of tiles that
			# are too far away and ignore them
			# Use an arbitrary radius to classify
			faraway_tiles = set ()

			for neighbour_id in sorted_field_ids :

				if field_id != neighbour_id :

					neighbour = field_dict [ neighbour_id ]

					if neighbour.tile_id not in faraway_tiles :

						dx = fx - neighbour.field_center [ 0 ]
						dy = fy - neighbour.field_center [ 1 ]
						d2 = dx * dx + dy * dy

						if d2 < n2 :

							# See if we want to add this neighbour
							if d2 < r2 :

								distance = math.sqrt ( d2 )

								field.neighbour_ids.append ( neighbour_id )
								field.neighbour_distances.append ( distance )

								sum = sum + 1

								out.write ( str ( field_id ) )
								out.write ( "," )
								out.write ( str ( neighbour_id ) )
								out.write ( "," )
								out.write ( str ( distance ) )
								out.write ( "\n" )
								out.flush ()

						else :

							# Tile too far away
							faraway_tiles.add ( neighbour.tile_id )

		# Done
		out.close ()

		print ( f"Done - recorded {sum} neighbours within a {neighbour_radius} radius, an average of {sum / len ( sorted_field_ids )} neighbours per field" )

	else :

		print ( f"Loading field neighbour information" )
		inp = open ( field_neighbours_fname )

		# Header
		inp.readline ()

		# Neighbours
		for line in inp :
			fields = line.strip ().split ( "," )
			field_id = int ( fields [ 0 ] )
			neighbour_id = int ( fields [ 1 ] )
			neighbour_distance = float ( fields [ 2 ] )

			field = field_dict [ field_id ]

			if field.neigbhour_ids is None :

				field.neighbour_ids = [ ]
				field.neighbour_distances = [ ]

			field.neighbour_ids.append ( neighbour_id )
			field.neighbour_distances.append ( neighbour_distance )

# Update
update_field_dict ()
update_overlap_fields ()
update_overlap_features ()
update_field_lookup ()
update_name_indices ()
update_data_support ()
update_field_neighbours ()
