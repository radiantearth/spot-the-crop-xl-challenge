# Start with simple model

# Then add overlap and neighbour distribution
# but base these on OOF predictions of simple
# model

import json
import math
import os
import random

import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

# Parameters

# Number of classes
n_class = 9

# Number of folds
n_folds = 10

# Models used
use_lgb = True
use_cat = True

# Number of iterations
n_lgb_iter = 1000
n_cat_iter = 10000

# Test code
test_only = False

if test_only :
	# Something light and fast
	n_lgb_iter = 20
	n_cat_iter = 20
	n_folds = 3

# Neighbourhood weight decay rates used
neighbour_decay = [ -1, -4 ]

# Working directory
work_dir = "../features/"

# Submission and oof directory
sub_dir = "../sub/"
os.makedirs ( sub_dir, exist_ok = True )

# Model submission files (=3x2x3 = 18!)
# For preliminary and dirty and final models (=3)
# For test and full data set (=2)
# For cat and lgb and combined (=3)

sub_prelim_test_lgb_fname = sub_dir + "l4-3pt.csv"
sub_prelim_all_lgb_fname = sub_dir + "l4-3pa.csv"
sub_prelim_test_cat_fname = sub_dir + "c4-3pt.csv"
sub_prelim_all_cat_fname = sub_dir + "c4-3pa.csv"
sub_prelim_test_both_fname = sub_dir + "b4-3pt.csv"
sub_prelim_all_both_fname = sub_dir + "b4-3pa.csv"

sub_dirty_oof_test_lgb_fname = sub_dir + "l4-3dt.csv"
sub_dirty_oof_all_lgb_fname = sub_dir + "l4-3da.csv"
sub_dirty_oof_test_cat_fname = sub_dir + "c4-3dt.csv"
sub_dirty_oof_all_cat_fname = sub_dir + "c4-3da.csv"
sub_dirty_oof_test_both_fname = sub_dir + "b4-3dt.csv"
sub_dirty_oof_all_both_fname = sub_dir + "b4-3da.csv"

sub_final_oof_test_lgb_fname = sub_dir + "l4-3ft.csv"
sub_final_oof_all_lgb_fname = sub_dir + "l4-3fa.csv"
sub_final_oof_test_cat_fname = sub_dir + "c4-3ft.csv"
sub_final_oof_all_cat_fname = sub_dir + "c4-3fa.csv"
sub_final_oof_test_both_fname = sub_dir + "b4-3ft.csv"
sub_final_oof_all_both_fname = sub_dir + "b4-3fa.csv"

sub_double_oof_test_lgb_fname = sub_dir + "l4-3ot.csv"
sub_double_oof_all_lgb_fname = sub_dir + "l4-3oa.csv"
sub_double_oof_test_cat_fname = sub_dir + "c4-3ot.csv"
sub_double_oof_all_cat_fname = sub_dir + "c4-3oa.csv"
sub_double_oof_test_both_fname = sub_dir + "b4-3ot.csv"
sub_double_oof_all_both_fname = sub_dir + "b4-3oa.csv"

# Do we claculate submission files for the clean version
do_clean = False

# Do we also calculate submission files for the double version
do_double = False

# Do we also calculate submission files for the dirty version
do_dirty = True

# Submission format - toggles between old and new format submissions
# https://zindi.africa/competitions/radiant-earth-spot-the-crop-xl-challenge/discussions/7402
sub_old_format = False

# Files
field_fname = work_dir + "fields.csv"
data_support_fname = work_dir + "data_support.npy"
overlap_fname = work_dir + "overlap_fields.csv"
neighbours_fname = work_dir + "neighbours.csv"
prelim_feature_matrix_fname = work_dir + "features4-3p.npy"
final_feature_matrix_fname = work_dir + "features4-3f.npy"
crop_labels_fname = "ref_south_africa_crops_competition_v1_train_labels/_common/raster_values.json"
field_labels_fname = "ref_south_africa_crops_competition_v1_train_labels/_common/field_info_train.csv"

# Utilities

# Utility to open, load, close and return a json file
def load_json_file ( fname ) :
	file = open ( fname )
	x = json.load ( file )
	file.close ()
	return x

# Set the random seed
def set_random_seed ( seed ) :
	np.random.seed ( seed )
	random.seed ( seed )

# Determine optimal weight between lgb and cat
def opt_lgb_cat ( lgb_matrix, cat_matrix ) :
	if use_lgb and use_cat :

		print ( f"\tOptimal mix of lgb and cat" )
		f = lgb_matrix

		# Start with lgb = 100%
		opt = log_loss ( train_labels, f [ train_rows ] )
		print ( f"\t\tlgb = 1 cat = 0 loss = {opt} ***" )

		# Loop in 5% increments
		for i in range ( 20 ) :

			w_lgb = 1 - 0.05 * (1 + i)
			w_cat = 1 - w_lgb
			t = (w_lgb * lgb_matrix + w_cat * cat_matrix) / (w_lgb + w_cat)
			l = log_loss ( train_labels, t [ train_rows ] )
			if l < opt :
				opt = l
				f = t
				print ( f"\t\tlgb = {w_lgb} cat = {w_cat} loss = {opt} ***" )
			else :
				print ( f"\t\tlgb = {w_lgb} cat = {w_cat} loss = {l}" )

	elif use_lgb :

		print ( f"\tOnly using lgb" )
		f = lgb_matrix

	elif use_cat :

		print ( f"\tOnly using cat" )
		f = cat_matrix

	return f

# Write a submission
def write_sub ( sub_fname, test_field_ids, pred ) :
	sub_out = open ( sub_fname, "w" )

	# Header
	sub_out.write ( "Field ID" )

	for crop in range ( n_class ) :
		sub_out.write ( ",Crop_ID_" )
		sub_out.write ( str ( crop + 1 ) )

	sub_out.write ( "\n" )

	# Predictions
	for row in range ( pred.shape [ 0 ] ) :
		sub_out.write ( str ( test_field_ids [ row ] ) )
		for p in pred [ row ] :
			sub_out.write ( "," )
			sub_out.write ( str ( p ) )
		sub_out.write ( "\n" )

	sub_out.close ()

# Read a submission file
def read_sub ( sub_fname ) :
	return np.loadtxt ( sub_fname, delimiter = ",", skiprows = 1, usecols = [ 1, 2, 3, 4, 5, 6, 7, 8, 9 ], dtype = float )

# Start

# This mirrors the support data structure
# Each row represents a single field
# Each date represents images taken during a given time slot, each roughly 15 days in length
# Each band represents a satellite band or composite band
n_rows = 122408
n_dates = 16
n_bands = 18

# Load crop labels
print ( "Reading crop labels" )
crop_labels = { }
for index, label in load_json_file ( crop_labels_fname ).items () :
	crop_labels [ label ] = int ( index )

# Fix up
# This fix up is required for the "old" format submissions
if sub_old_format :

	del crop_labels [ "No Data" ]

	crop_label_keys = sorted ( crop_labels.keys () )

	for i, label in enumerate ( crop_label_keys ) :

		crop_labels [ label ] = i + 1

	crop_labels [ "No Data" ] = 0

# This now contains field_id, train (1 or 0) and tile_id
print ( "Reading fields" )
field_ids = np.loadtxt ( field_fname, dtype = int, delimiter = ",", skiprows = 1, usecols = (0, 1, 2) )

# Build a lookup from the field ids
# First col = field_id of row
# Second col = row of field_id
# Third col = label (0 if in test)
# Fourth column = 1 if in training and 0 if in testing set
# Fifth column = tile_id of row
# Sixth column = test row of field_id if in test
print ( f"\tCreating lookup" )

lookup = np.zeros ( [ field_ids.max () + 1, 6 ], dtype = int )
lookup.fill ( -1 )

print ( f"\tField ID and row mappings" )

row_test = 0

for row in range ( n_rows ) :

	field_id = field_ids [ row, 0 ]
	in_train = field_ids [ row, 1 ]
	tile_id = field_ids [ row, 2 ]

	lookup [ row, 0 ] = field_id
	lookup [ field_id, 1 ] = row
	lookup [ row, 3 ] = in_train
	lookup [ row, 4 ] = tile_id

	if in_train == 0 :

		lookup [ row, 5 ] = row_test
		row_test = row_test + 1

# Load field labels and store in third col
print ( f"\tLoading known labels" )
fin = open ( field_labels_fname )
fin.readline ()

for line in fin :

	fields = line.strip ().split ( "," )
	field_id = int ( fields [ 0 ] )
	label = crop_labels [ fields [ 1 ] ]
	row = lookup [ field_id, 1 ]
	lookup [ row, 2 ] = label

fin.close ()

# We will have avg, std (= 2) for each of band x used date
n_sat_features = n_dates * n_bands * 2

# Other features are pixels, size, fragmentation, perimeter, perimeter to size, circular measure (= 6)
n_other_features = 6

n_total_features = n_sat_features + n_other_features

# Useful for later
all_field_ids = lookup [ :n_rows, 0 ]

train_rows = lookup [ : n_rows, 3 ] == 1

train_labels = lookup [ : n_rows, 2 ] [ train_rows ] - 1
train_field_ids = lookup [ : n_rows, 0 ] [ train_rows ]
train_tile_ids = lookup [ : n_rows, 4 ] [ train_rows ]

test_field_ids = lookup [ :n_rows, 0 ] [ ~train_rows ]

# See if the matrix is already available
if os.path.isfile ( prelim_feature_matrix_fname ) :

	print ( "Reading preliminary feature matrix" )

	# Load tile overlap offsets
	data_result_matrix = np.load ( prelim_feature_matrix_fname )

else :

	# Load the support data
	print ( "Reading support data" )
	data_support_matrix = np.load ( data_support_fname )

	n_rows = data_support_matrix.shape [ 0 ]
	n_dates = data_support_matrix.shape [ 1 ]
	n_bands = data_support_matrix.shape [ 2 ]

	print ( f"\tThere are {n_rows} rows and {n_dates} dates and {n_bands} bands available" )

	# Create a matrix to house all the features
	data_result_matrix = np.zeros ( [ n_rows, n_total_features ], dtype = float )

	# Now calculate the satellite image features from the support matrix
	print ( f"Calculating image features" )

	for row in range ( n_rows ) :

		for date in range ( n_dates ) :

			for band in range ( n_bands ) :

				# Extract data

				# nobs, min, max, sum, sum of squares
				n = data_support_matrix [ row, date, band, 0 ]

				if n > 0 :

					mn = data_support_matrix [ row, date, band, 1 ]
					mx = data_support_matrix [ row, date, band, 2 ]
					s1 = data_support_matrix [ row, date, band, 3 ]
					s2 = data_support_matrix [ row, date, band, 4 ]

					# Load avg and std
					data_result_matrix [ row, 2 * (date * n_bands + band) + 1 ] = s1 / n

					if n > 1 :

						t = n * s2 - s1 * s1

						# Check for rounding error
						if t > 0 :

							data_result_matrix [ row, 2 * (date * n_bands + band) + 1 ] = math.sqrt ( t / (n * (n - 1)) )

	# Load the size, fragmentation, perimeter, petrimeter to size and circular measure from a file
	print ( f"Loading other features" )

	data_result_matrix [ :, n_sat_features : n_sat_features + n_other_features ] = np.loadtxt ( field_fname, dtype = float, delimiter = ",", skiprows = 1, usecols = (3, 4, 7, 8, 9, 10) )

	# Write out the feature matrix
	print ( "Saving preliminary feature matrix" )
	np.save ( prelim_feature_matrix_fname, data_result_matrix )

# Now we can model
# The first model is used to classify OOF fields
# and this is later used to construct overlap and
# neighbour probabilities

# See if preliminary results are already available
if os.path.isfile ( sub_prelim_all_lgb_fname ) and os.path.isfile ( sub_prelim_all_cat_fname ) :

	print ( "Reading preliminary predictions" )

	print ( "\tlgb" )
	pred_lgb_matrix = read_sub ( sub_prelim_all_lgb_fname )

	print ( "\tcat" )
	pred_cat_matrix = read_sub ( sub_prelim_all_cat_fname )

else :

	print ( "Fitting preliminary model" )

	rs = 20210831

	set_random_seed ( rs )

	train_matrix = data_result_matrix [ train_rows ]
	test_matrix = data_result_matrix [ ~ train_rows ]

	# Predictions, both OOF and for test
	pred_lgb_matrix = np.zeros ( [ data_result_matrix.shape [ 0 ], n_class ], dtype = float )
	pred_cat_matrix = np.zeros ( [ data_result_matrix.shape [ 0 ], n_class ], dtype = float )

	pred_lgb_oof_matrix = np.zeros ( [ train_matrix.shape [ 0 ], n_class ], dtype = float )
	pred_cat_oof_matrix = np.zeros ( [ train_matrix.shape [ 0 ], n_class ], dtype = float )

	skf = GroupKFold ( n_splits = n_folds )

	for fold, (itrain, ivalid) in enumerate ( skf.split ( train_matrix, train_labels, train_tile_ids ) ) :

		print ( f"\t----[ Fold {fold + 1}/{n_folds} ]{'-' * 100}" )

		train_matrix_fold = train_matrix [ itrain ]
		eval_matrix_fold = train_matrix [ ivalid ]
		train_labels_fold = train_labels [ itrain ]
		eval_labels_fold = train_labels [ ivalid ]

		if use_lgb :

			print ( f"\t\tFitting lgb" )
			lgb_params = { "objective"      : "multiclass",
				       "num_class"      : n_class,
				       "bagging_seed"   : rs + 7 + fold * 53,
				       "force_col_wise" : True,
				       "verbosity"      : 0 }
			lgb_train = lgb.Dataset ( train_matrix_fold, train_labels_fold )
			lgb_eval = lgb.Dataset ( eval_matrix_fold, eval_labels_fold )
			lgb_clf = lgb.train ( lgb_params, lgb_train, n_lgb_iter, valid_sets = [ lgb_eval ], early_stopping_rounds = 150, verbose_eval = False )
			print ( f"\t\t\t{lgb_clf.best_score [ 'valid_0' ] [ 'multi_logloss' ]}" )

			# Now predict for test
			pred_lgb_matrix [ ~ train_rows ] = pred_lgb_matrix [ ~ train_rows ] + lgb_clf.predict ( test_matrix ) / n_folds

			# Also predict OOF
			pred_lgb_oof_matrix [ ivalid ] = lgb_clf.predict ( eval_matrix_fold )

		if use_cat :

			print ( f"\t\tFitting cat" )
			cat_clf = CatBoostClassifier ( iterations = n_cat_iter, classes_count = n_class, random_state = rs + 11 + fold * 97, verbose = False )
			cat_clf.fit ( train_matrix_fold, train_labels_fold, eval_set = (eval_matrix_fold, eval_labels_fold) )
			print ( f"\t\t\t{cat_clf.get_best_score () [ 'validation' ] [ 'MultiClass' ]}" )

			# Now predict for test
			pred_cat_matrix [ ~ train_rows ] = pred_cat_matrix [ ~ train_rows ] + cat_clf.predict_proba ( test_matrix ) / n_folds

			# Also predict OOF
			pred_cat_oof_matrix [ ivalid ] = cat_clf.predict_proba ( eval_matrix_fold )

	# Load OOF predictions into results
	pred_lgb_matrix [ train_rows ] = pred_lgb_oof_matrix
	pred_cat_matrix [ train_rows ] = pred_cat_oof_matrix

	# Writing preliminary result
	print ( "\tWriting preliminary results as submission file" )

	# We save both, even if we did just one
	# since we check for both files when we
	# test

	print ( "\t\tlgb" )
	write_sub ( sub_prelim_all_lgb_fname, all_field_ids, pred_lgb_matrix )
	write_sub ( sub_prelim_test_lgb_fname, test_field_ids, pred_lgb_matrix [ ~train_rows ] )

	print ( "\t\tcat" )
	write_sub ( sub_prelim_all_cat_fname, all_field_ids, pred_cat_matrix )
	write_sub ( sub_prelim_test_cat_fname, test_field_ids, pred_cat_matrix [ ~train_rows ] )

	# Done
	print ( "~" * 100 )
	print ( "Done" )

# Apply weights
pred_prelim_matrix = opt_lgb_cat ( pred_lgb_matrix, pred_cat_matrix )

write_sub ( sub_prelim_all_both_fname, all_field_ids, pred_prelim_matrix )
write_sub ( sub_prelim_test_both_fname, test_field_ids, pred_prelim_matrix [ ~train_rows ] )

# Report preliminary results
print ( f"Preliminary model results" )
print ( f"OOF" )

if use_lgb :

	print ( f"\tlgb  : {log_loss ( train_labels, pred_lgb_matrix [ train_rows ] )}" )

if use_cat :

	print ( f"\tcat  : {log_loss ( train_labels, pred_cat_matrix [ train_rows ] )}" )

print ( f"\tboth : {log_loss ( train_labels, pred_prelim_matrix [ train_rows ] )}" )

# Now we can load the overlap and the neighbour information
# This is very similar to the preliminary step
# We have the same features as before but this time we add
# new features, being the overlap between fields as well
# as the distribution of neighbouring fields

# As before we will have avg, std (= 2) for each of band x used date
n_sat_features = n_dates * n_bands * 2

# As before, other features are pixels, size, fragmentation, perimeter, perimeter to size, circular measure (= 6)
n_other_features = 6

# This time we add neighbour distribution features
n_neighbour_features = n_class * len ( neighbour_decay )

# Finally we will add field overlap features
# This includes the distribution of labels in the overlap region (=n_class)
# followed by some border and overlap specific features (=8 border size to size, overlap size to border size, combined pixel size, combined size, combined fragmentation, combined perimeter, combined perimeter to size, combined circular measure)
n_overlap_features = n_class + 8

n_total_features = n_sat_features + n_other_features + n_neighbour_features + n_overlap_features

# This function will prepare and return a feature matrix
# It can base the features either on actual values (dirty)
# or on OOF predictions (clean) and will NOT use any
# features currently in the hold out set.
# The clean parameter is either None for the dirty
# version or equal to a matrix that holds the OOF
# predictions for a clean version
def make_feature_matrix ( clean, train ) :
	# The return matrix
	print ( "\tMaking feature matrix" )
	feature_matrix = np.zeros ( [ n_rows, n_total_features ], dtype = float )

	# The first few columns are the same as the columns of the preliminary model
	print ( "\t\tSatellite and other features" )
	feature_matrix [ :, :n_sat_features + n_other_features ] = np.load ( prelim_feature_matrix_fname )

	# Now calculate the neighbour distributions
	print ( "\t\tNeighbourhood distributions" )

	# Previous field id and sum of weights
	pid = 0
	wsm = [ 0 for i in neighbour_decay ]

	# Start of neighbour features / end of other features
	I = n_sat_features + n_other_features

	fin = open ( neighbours_fname )

	# Header
	fin.readline ()

	for line in fin :

		fields = line.strip ().split ( "," )
		field_id = int ( fields [ 0 ] )

		# If we have a new ID first adjust the old one
		if pid != field_id and pid > 0 :

			# Adjust for the previous row
			for i, w in enumerate ( wsm ) :

				if w > 0 :

					feature_matrix [ row, I + i * n_class :I + (i + 1) * n_class ] = feature_matrix [ row, I + i * n_class :I + (i + 1) * n_class ] / w
					wsm [ i ] = 0

		# Evaluate this (next) row
		row = lookup [ field_id, 1 ]
		neighbour_id = int ( fields [ 1 ] )
		distance = float ( fields [ 2 ] )
		neighbour_row = lookup [ neighbour_id, 1 ]

		# We only "know" about the neighbour if it is in the train sample
		if train [ neighbour_row ] :

			if clean is None :

				# The dirty version uses the actual label
				neighbour_label = lookup [ neighbour_row, 2 ]

				# See if we have a valid label from training
				if neighbour_label > 0 :

					for i, w in enumerate ( neighbour_decay ) :

						weight = math.pow ( distance, w )
						feature_matrix [ row, I + i * n_class + neighbour_label - 1 ] = feature_matrix [ row, I + i * n_class + neighbour_label - 1 ] + weight
						wsm [ i ] = wsm [ i ] + weight

			else :

				# The clean version uses the OOF predictions
				for i, w in enumerate ( neighbour_decay ) :

					weight = math.pow ( distance, w )
					feature_matrix [ row, I + i * n_class :I + (i + 1) * n_class ] = feature_matrix [ row, I + i * n_class :I + (i + 1) * n_class ] + weight * clean [ row ]
					wsm [ i ] = wsm [ i ] + weight

		# Move to next row
		pid = field_id

	fin.close ()

	# Adjust final row
	for i, w in enumerate ( wsm ) :

		if w > 0 :

			feature_matrix [ row, I + i * n_class :I + (i + 1) * n_class ] = feature_matrix [ row, I + i * n_class :I + (i + 1) * n_class ] / w

	# Next load overlap

	# This support class will help with calculations
	class Overlap :
		pixel_size = None
		border_size = None
		# Combined
		pixel_size_combined = None
		field_size = None
		center_x = None
		center_y = None
		fragmentation = None
		perimeter = None
		perimeter_to_size = None
		circular_measure = None
		label_counts = None
		# Row in prediction matrix
		pred_row = None

	# For each field id, store it in here
	overlap_dict = { }

	print ( "\t\tCalculating field overlap information" )

	# We need the field pixel size to proceed
	field_pixels = np.loadtxt ( field_fname, dtype = int, delimiter = ",", skiprows = 1, usecols = 3 )

	# Now read the overlap information
	fin = open ( overlap_fname )

	fin.readline ()

	for line in fin :

		fields = line.strip ().split ( "," )

		field_id = int ( fields [ 0 ] )
		border_pixels = int ( fields [ 1 ] )
		overlap_id = int ( fields [ 2 ] )
		overlap_count = int ( fields [ 3 ] )

		# Where are these fields located in the data matrix
		row = lookup [ field_id, 1 ]

		# If overlap id == 0 this just means the field is found in the tile border
		if overlap_id > 0 :

			# The row of the overlapping field
			overlap_row = lookup [ overlap_id, 1 ]

			# Only use info from training set
			if train [ overlap_row ] :

				if clean is None :

					# The dirty version uses the actual label
					overlap_label = lookup [ overlap_row, 2 ]

					# Only note overlap with known labels
					if overlap_label > 0 :

						overlap = overlap_dict.get ( field_id, None )

						if overlap is None :

							# Create a new overlap
							overlap = Overlap ()

							# Note
							# Other features are pixels, size, fragmentation, perimeter, perimeter to size, circular measure
							# And I points to end of other features, so I-1 - circular measure e.g.

							overlap.pixel_size = field_pixels [ row ]
							overlap.border_size = border_pixels
							overlap.pixel_size_combined = field_pixels [ row ]
							overlap.field_size = feature_matrix [ row, I - 7 ]
							overlap.center_x = feature_matrix [ row, I - 6 ]
							overlap.center_y = feature_matrix [ row, I - 5 ]
							overlap.fragmentation = feature_matrix [ row, I - 4 ]
							overlap.perimeter = feature_matrix [ row, I - 3 ]
							overlap.perimeter_to_size = feature_matrix [ row, I - 2 ]
							overlap.circular_measure = feature_matrix [ row, I - 1 ]

							overlap.label_counts = np.zeros ( n_class, dtype = int )

							# Store the overlap for this field
							overlap_dict [ field_id ] = overlap

						# Now update

						# Weights
						p0 = overlap.pixel_size
						p1 = field_pixels [ overlap_row ]
						pt = p0 + p1
						f0 = p0 / pt
						f1 = p1 / pt

						# Update
						overlap.pixel_size_combined = f0 * overlap.pixel_size_combined + f1 * p1
						overlap.field_size = f0 * overlap.field_size + f1 * feature_matrix [ overlap_row, I - 7 ]
						overlap.center_x = f0 * overlap.center_x + f1 * feature_matrix [ overlap_row, I - 6 ]
						overlap.center_y = f0 * overlap.center_y + f1 * feature_matrix [ overlap_row, I - 5 ]
						overlap.fragmentation = f0 * overlap.fragmentation + f1 * feature_matrix [ overlap_row, I - 4 ]
						overlap.perimeter = f0 * overlap.perimeter + f1 * feature_matrix [ overlap_row, I - 3 ]
						overlap.perimeter_to_size = f0 * overlap.perimeter_to_size + f1 * feature_matrix [ overlap_row, I - 2 ]
						overlap.circular_measure = f0 * overlap.circular_measure + f1 * feature_matrix [ overlap_row, I - 1 ]

						# Label count
						overlap.label_counts [ overlap_label - 1 ] = overlap.label_counts [ overlap_label - 1 ] + overlap_count

				else :


					# For the clean version we will use the OOF prediction of the field
					overlap = overlap_dict.get ( field_id, None )

					if overlap is None :

						# Create new overlap
						overlap = Overlap ()

						# Note
						# Other features are pixels, size, fragmentation, perimeter, perimeter to size, circular measure
						# And I points to end of other features, so I-1 - circular measure e.g.

						overlap.pixel_size = field_pixels [ row ]
						overlap.border_size = border_pixels
						overlap.pixel_size_combined = field_pixels [ row ]
						overlap.field_size = feature_matrix [ row, I - 7 ]
						overlap.center_x = feature_matrix [ row, I - 6 ]
						overlap.center_y = feature_matrix [ row, I - 5 ]
						overlap.fragmentation = feature_matrix [ row, I - 4 ]
						overlap.perimeter = feature_matrix [ row, I - 3 ]
						overlap.perimeter_to_size = feature_matrix [ row, I - 2 ]
						overlap.circular_measure = feature_matrix [ row, I - 1 ]

						# Since this time we work with probs, not actual field lables
						# we change the type to float (previously it was int)
						overlap.label_counts = np.zeros ( n_class, dtype = float )

						# Store new overlap
						overlap_dict [ field_id ] = overlap

					# Now update

					# Weights
					p0 = overlap.pixel_size
					p1 = field_pixels [ overlap_row ]
					pt = p0 + p1
					f0 = p0 / pt
					f1 = p1 / pt

					# Update
					overlap.pixel_size_combined = f0 * overlap.pixel_size_combined + f1 * p1
					overlap.field_size = f0 * overlap.field_size + f1 * feature_matrix [ overlap_row, I - 7 ]
					overlap.center_x = f0 * overlap.center_x + f1 * feature_matrix [ overlap_row, I - 6 ]
					overlap.center_y = f0 * overlap.center_y + f1 * feature_matrix [ overlap_row, I - 5 ]
					overlap.fragmentation = f0 * overlap.fragmentation + f1 * feature_matrix [ overlap_row, I - 4 ]
					overlap.perimeter = f0 * overlap.perimeter + f1 * feature_matrix [ overlap_row, I - 3 ]
					overlap.perimeter_to_size = f0 * overlap.perimeter_to_size + f1 * feature_matrix [ overlap_row, I - 2 ]
					overlap.circular_measure = f0 * overlap.circular_measure + f1 * feature_matrix [ overlap_row, I - 1 ]

					# Label count
					# This time the label is based on the OOF predictions, not the actual field labels
					# Note that, as before, the weight is the number of pixels that overlap or the overlap count
					overlap.label_counts = overlap.label_counts + overlap_count * clean [ overlap_row ]

	fin.close ()

	# Indexing
	overlap_field_ids = sorted ( overlap_dict.keys () )

	# Since some fields do not have overlap, this is NAN by default
	I = I + n_neighbour_features
	feature_matrix [ :, I :I + n_overlap_features ] = np.nan

	# Load the calculated overlap
	for i, field_id in enumerate ( overlap_field_ids ) :

		row = lookup [ field_id, 1 ]
		overlap = overlap_dict [ field_id ]

		if overlap.border_size > 0 :

			feature_matrix [ row, I :I + n_class ] = overlap.label_counts / overlap.border_size
			feature_matrix [ row, I + n_class ] = overlap.border_size / overlap.pixel_size
			feature_matrix [ row, I + n_class + 1 ] = overlap.label_counts.sum () / overlap.border_size
			feature_matrix [ row, I + n_class + 2 ] = overlap.pixel_size_combined
			feature_matrix [ row, I + n_class + 3 ] = overlap.field_size
			feature_matrix [ row, I + n_class + 4 ] = overlap.fragmentation
			feature_matrix [ row, I + n_class + 5 ] = overlap.perimeter
			feature_matrix [ row, I + n_class + 6 ] = overlap.perimeter_to_size
			feature_matrix [ row, I + n_class + 7 ] = overlap.circular_measure

	# Done
	print ( "\tFeatures ready" )
	return feature_matrix

# Fit a model to the clean features
if do_clean :

	# Now we can fit a model to the final, OOF predictions based (clean) feature matrix
	if os.path.isfile ( sub_final_oof_all_lgb_fname ) and os.path.isfile ( sub_final_oof_all_cat_fname ) :

		print ( "Reading final / clean model predictions" )

		print ( "\tlgb" )
		pred_lgb_matrix = read_sub ( sub_final_oof_all_lgb_fname )

		print ( "\tcat" )
		pred_cat_matrix = read_sub ( sub_final_oof_all_cat_fname )

	else :

		print ( "Fitting final / clean model" )

		rs = 20210902

		set_random_seed ( rs )

		# Predictions, both OOF and for test
		pred_lgb_matrix = np.zeros ( [ n_rows, n_class ], dtype = float )
		pred_cat_matrix = np.zeros ( [ n_rows, n_class ], dtype = float )

		pred_lgb_oof_matrix = np.zeros ( [ len ( train_labels ), n_class ], dtype = float )
		pred_cat_oof_matrix = np.zeros ( [ len ( train_labels ), n_class ], dtype = float )

		# The preliminary model was built using OOF features so we use all
		# of its rows here and need not exclude any when training below
		if os.path.isfile ( final_feature_matrix_fname ) :

			print ( "\tLoading clean feature matrix" )
			feature_matrix = np.load ( final_feature_matrix_fname )

		else :

			# True for all indices
			itrue = np.ones ( n_rows, dtype = bool )
			feature_matrix = make_feature_matrix ( pred_prelim_matrix, itrue )

			# Write out the feature matrix
			print ( "\tSaving clean feature matrix" )
			np.save ( final_feature_matrix_fname, feature_matrix )

		# Later on we will also exclude the validation set's rows in the
		# "double" version of the model
		iall = np.arange ( n_rows, dtype = int )
		i0 = iall [ train_rows ]

		skf = GroupKFold ( n_splits = n_folds )

		for fold, (itrain, ivalid) in enumerate ( skf.split ( train_labels, train_labels, train_tile_ids ) ) :

			print ( f"\t----[ Fold {fold + 1}/{n_folds} ]{'-' * 100}" )

			train_matrix_fold = feature_matrix [ i0 [ itrain ] ]
			eval_matrix_fold = feature_matrix [ i0 [ ivalid ] ]
			train_labels_fold = train_labels [ itrain ]
			eval_labels_fold = train_labels [ ivalid ]

			if use_lgb :

				print ( f"\t\tFitting lgb" )
				lgb_params = { "objective"      : "multiclass",
					       "num_class"      : n_class,
					       "bagging_seed"   : rs + 19 + fold * 41,
					       "force_col_wise" : True,
					       "verbosity"      : 0 }
				lgb_train = lgb.Dataset ( train_matrix_fold, train_labels_fold )
				lgb_eval = lgb.Dataset ( eval_matrix_fold, eval_labels_fold )
				lgb_clf = lgb.train ( lgb_params, lgb_train, n_lgb_iter, valid_sets = [ lgb_eval ], early_stopping_rounds = 150, verbose_eval = False )
				print ( f"\t\t\t{lgb_clf.best_score [ 'valid_0' ] [ 'multi_logloss' ]}" )

				# Now predict for test
				pred_lgb_matrix [ ~ train_rows ] = pred_lgb_matrix [ ~ train_rows ] + lgb_clf.predict ( feature_matrix [ ~train_rows ] ) / n_folds

				# Also predict OOF
				pred_lgb_oof_matrix [ ivalid ] = lgb_clf.predict ( eval_matrix_fold )

			if use_cat :

				print ( f"\t\tFitting cat" )
				cat_clf = CatBoostClassifier ( iterations = n_cat_iter, classes_count = n_class, random_state = rs + 23 + fold * 101, verbose = False )
				cat_clf.fit ( train_matrix_fold, train_labels_fold, eval_set = (eval_matrix_fold, eval_labels_fold) )
				print ( f"\t\t\t{cat_clf.get_best_score () [ 'validation' ] [ 'MultiClass' ]}" )

				# Now predict for test
				pred_cat_matrix [ ~ train_rows ] = pred_cat_matrix [ ~ train_rows ] + cat_clf.predict_proba ( feature_matrix [ ~train_rows ] ) / n_folds

				# Also predict OOF
				pred_cat_oof_matrix [ ivalid ] = cat_clf.predict_proba ( eval_matrix_fold )

		# Load OOF predictions into results
		pred_lgb_matrix [ train_rows ] = pred_lgb_oof_matrix
		pred_cat_matrix [ train_rows ] = pred_cat_oof_matrix

		# Writing preliminary result
		print ( "\tWriting final results as submission file" )

		print ( "\t\tlgb" )
		write_sub ( sub_final_oof_all_lgb_fname, all_field_ids, pred_lgb_matrix )
		write_sub ( sub_final_oof_test_lgb_fname, test_field_ids, pred_lgb_matrix [ ~train_rows ] )

		print ( "\t\tcat" )
		write_sub ( sub_final_oof_all_cat_fname, all_field_ids, pred_cat_matrix )
		write_sub ( sub_final_oof_test_cat_fname, test_field_ids, pred_cat_matrix [ ~train_rows ] )

		# Done
		print ( "~" * 100 )
		print ( "Done" )

	# Now combine the final predictions
	pred_clean_matrix = opt_lgb_cat ( pred_lgb_matrix, pred_cat_matrix )
	write_sub ( sub_final_oof_all_both_fname, all_field_ids, pred_clean_matrix )
	write_sub ( sub_final_oof_test_both_fname, test_field_ids, pred_clean_matrix [ ~train_rows ] )

	# Report final results
	print ( f"Final / clean model results" )
	print ( f"OOF" )

	if use_lgb :

		print ( f"\tlgb  : {log_loss ( train_labels, pred_lgb_matrix [ train_rows ] )}" )

	if use_cat :

		print ( f"\tcat  : {log_loss ( train_labels, pred_cat_matrix [ train_rows ] )}" )

	print ( f"\tboth : {log_loss ( train_labels, pred_clean_matrix [ train_rows ] )}" )

# Now fit a model to the double features
# These are called double as we exclude
# OOF features both in the preliminary
# and in the final model, so we doubly
# exclude features
if do_double :

	if os.path.isfile ( sub_double_oof_all_lgb_fname ) and os.path.isfile ( sub_double_oof_all_cat_fname ) :

		print ( "Reading final / double model predictions" )

		print ( "\tlgb" )
		pred_lgb_matrix = read_sub ( sub_double_oof_all_lgb_fname )

		print ( "\tcat" )
		pred_cat_matrix = read_sub ( sub_double_oof_all_cat_fname )

	else :

		print ( "Fitting double model" )

		rs = 20210915

		set_random_seed ( rs )

		# Predictions, both OOF and for test
		pred_lgb_matrix = np.zeros ( [ n_rows, n_class ], dtype = float )
		pred_cat_matrix = np.zeros ( [ n_rows, n_class ], dtype = float )

		pred_lgb_oof_matrix = np.zeros ( [ len ( train_labels ), n_class ], dtype = float )
		pred_cat_oof_matrix = np.zeros ( [ len ( train_labels ), n_class ], dtype = float )

		iall = np.arange ( n_rows, dtype = int )
		i0 = iall [ train_rows ]

		skf = GroupKFold ( n_splits = n_folds )

		for fold, (itrain, ivalid) in enumerate ( skf.split ( train_labels, train_labels, train_tile_ids ) ) :

			print ( f"\t----[ Fold {fold + 1}/{n_folds} ]{'-' * 100}" )

			# Only use valid training set rows
			# This is the second time (the
			# "double" time) that we exclude
			# oof features from the model
			train = np.ones ( n_rows, dtype = bool )
			train [ iall [ ivalid ] ] = False

			# Get "double" feature matrix based on excluding validation set rows
			feature_matrix = make_feature_matrix ( pred_prelim_matrix, train )

			train_matrix_fold = feature_matrix [ i0 [ itrain ] ]
			eval_matrix_fold = feature_matrix [ i0 [ ivalid ] ]
			train_labels_fold = train_labels [ itrain ]
			eval_labels_fold = train_labels [ ivalid ]

			if use_lgb :

				print ( f"\t\tFitting lgb" )
				lgb_params = { "objective"      : "multiclass",
					       "num_class"      : n_class,
					       "bagging_seed"   : rs + 19 + fold * 41,
					       "force_col_wise" : True,
					       "verbosity"      : 0 }
				lgb_train = lgb.Dataset ( train_matrix_fold, train_labels_fold )
				lgb_eval = lgb.Dataset ( eval_matrix_fold, eval_labels_fold )
				lgb_clf = lgb.train ( lgb_params, lgb_train, n_lgb_iter, valid_sets = [ lgb_eval ], early_stopping_rounds = 150, verbose_eval = False )
				print ( f"\t\t\t{lgb_clf.best_score [ 'valid_0' ] [ 'multi_logloss' ]}" )

				# Now predict for test
				pred_lgb_matrix [ ~ train_rows ] = pred_lgb_matrix [ ~ train_rows ] + lgb_clf.predict ( feature_matrix [ ~train_rows ] ) / n_folds

				# Also predict OOF
				pred_lgb_oof_matrix [ ivalid ] = lgb_clf.predict ( eval_matrix_fold )

			if use_cat :

				print ( f"\t\tFitting cat" )
				cat_clf = CatBoostClassifier ( iterations = n_cat_iter, classes_count = n_class, random_state = rs + 23 + fold * 101, verbose = False )
				cat_clf.fit ( train_matrix_fold, train_labels_fold, eval_set = (eval_matrix_fold, eval_labels_fold) )
				print ( f"\t\t\t{cat_clf.get_best_score () [ 'validation' ] [ 'MultiClass' ]}" )

				# Now predict for test
				pred_cat_matrix [ ~ train_rows ] = pred_cat_matrix [ ~ train_rows ] + cat_clf.predict_proba ( feature_matrix [ ~train_rows ] ) / n_folds

				# Also predict OOF
				pred_cat_oof_matrix [ ivalid ] = cat_clf.predict_proba ( eval_matrix_fold )

		# Load double OOF predictions into results
		pred_lgb_matrix [ train_rows ] = pred_lgb_oof_matrix
		pred_cat_matrix [ train_rows ] = pred_cat_oof_matrix

		# Writing double result
		print ( "\tWriting double results as submission file" )

		print ( "\t\tlgb" )
		write_sub ( sub_double_oof_all_lgb_fname, all_field_ids, pred_lgb_matrix )
		write_sub ( sub_double_oof_test_lgb_fname, test_field_ids, pred_lgb_matrix [ ~train_rows ] )

		print ( "\t\tcat" )
		write_sub ( sub_double_oof_all_cat_fname, all_field_ids, pred_cat_matrix )
		write_sub ( sub_double_oof_test_cat_fname, test_field_ids, pred_cat_matrix [ ~train_rows ] )

		# Done
		print ( "~" * 100 )
		print ( "Done" )

	# Now combine the double predictions
	pred_double_matrix = opt_lgb_cat ( pred_lgb_matrix, pred_cat_matrix )
	write_sub ( sub_double_oof_all_both_fname, all_field_ids, pred_double_matrix )
	write_sub ( sub_double_oof_test_both_fname, test_field_ids, pred_double_matrix [ ~train_rows ] )

	# Report double results
	print ( f"Double model results" )
	print ( f"OOF" )

	if use_lgb :

		print ( f"\tlgb  : {log_loss ( train_labels, pred_lgb_matrix [ train_rows ] )}" )

	if use_cat :

		print ( f"\tcat  : {log_loss ( train_labels, pred_cat_matrix [ train_rows ] )}" )

	print ( f"\tboth : {log_loss ( train_labels, pred_double_matrix [ train_rows ] )}" )

# Now fit a model to the dirty features
if do_dirty :

	if os.path.isfile ( sub_dirty_oof_all_lgb_fname ) and os.path.isfile ( sub_dirty_oof_all_cat_fname ) :

		print ( "Reading final / dirty model predictions" )

		print ( "\tlgb" )

		pred_lgb_matrix = read_sub ( sub_dirty_oof_all_lgb_fname )

		print ( "\tcat" )

		pred_cat_matrix = read_sub ( sub_dirty_oof_all_cat_fname )

	else :

		print ( "Fitting dirty model" )

		rs = 20210930

		set_random_seed ( rs )

		# Predictions, both OOF and for test
		pred_lgb_matrix = np.zeros ( [ n_rows, n_class ], dtype = float )
		pred_cat_matrix = np.zeros ( [ n_rows, n_class ], dtype = float )

		pred_lgb_oof_matrix = np.zeros ( [ len ( train_labels ), n_class ], dtype = float )
		pred_cat_oof_matrix = np.zeros ( [ len ( train_labels ), n_class ], dtype = float )

		iall = np.arange ( n_rows, dtype = int )
		i0 = iall [ train_rows ]

		skf = GroupKFold ( n_splits = n_folds )

		for fold, (itrain, ivalid) in enumerate ( skf.split ( train_labels, train_labels, train_tile_ids ) ) :

			print ( f"\t----[ Fold {fold + 1}/{n_folds} ]{'-' * 100}" )

			# Only use valid training set rows
			train = np.ones ( n_rows, dtype = bool )
			train [ iall [ ivalid ] ] = False

			# Get dirty feature matrix
			feature_matrix = make_feature_matrix ( None, train )

			train_matrix_fold = feature_matrix [ i0 [ itrain ] ]
			eval_matrix_fold = feature_matrix [ i0 [ ivalid ] ]
			train_labels_fold = train_labels [ itrain ]
			eval_labels_fold = train_labels [ ivalid ]

			if use_lgb :

				print ( f"\t\tFitting lgb" )
				lgb_params = { "objective"      : "multiclass",
					       "num_class"      : n_class,
					       "bagging_seed"   : rs + 19 + fold * 41,
					       "force_col_wise" : True,
					       "verbosity"      : 0 }
				lgb_train = lgb.Dataset ( train_matrix_fold, train_labels_fold )
				lgb_eval = lgb.Dataset ( eval_matrix_fold, eval_labels_fold )
				lgb_clf = lgb.train ( lgb_params, lgb_train, n_lgb_iter, valid_sets = [ lgb_eval ], early_stopping_rounds = 150, verbose_eval = False )
				print ( f"\t\t\t{lgb_clf.best_score [ 'valid_0' ] [ 'multi_logloss' ]}" )

				# Now predict for test
				pred_lgb_matrix [ ~ train_rows ] = pred_lgb_matrix [ ~ train_rows ] + lgb_clf.predict ( feature_matrix [ ~train_rows ] ) / n_folds

				# Also predict OOF
				pred_lgb_oof_matrix [ ivalid ] = lgb_clf.predict ( eval_matrix_fold )

			if use_cat :

				print ( f"\t\tFitting cat" )
				cat_clf = CatBoostClassifier ( iterations = n_cat_iter, classes_count = n_class, random_state = rs + 23 + fold * 101, verbose = False )
				cat_clf.fit ( train_matrix_fold, train_labels_fold, eval_set = (eval_matrix_fold, eval_labels_fold) )
				print ( f"\t\t\t{cat_clf.get_best_score () [ 'validation' ] [ 'MultiClass' ]}" )

				# Now predict for test
				pred_cat_matrix [ ~ train_rows ] = pred_cat_matrix [ ~ train_rows ] + cat_clf.predict_proba ( feature_matrix [ ~train_rows ] ) / n_folds

				# Also predict OOF
				pred_cat_oof_matrix [ ivalid ] = cat_clf.predict_proba ( eval_matrix_fold )

		# Load dirty OOF predictions into results
		pred_lgb_matrix [ train_rows ] = pred_lgb_oof_matrix
		pred_cat_matrix [ train_rows ] = pred_cat_oof_matrix

		# Writing dirty result
		print ( "\tWriting dirty results as submission file" )

		print ( "\t\tlgb" )
		write_sub ( sub_dirty_oof_all_lgb_fname, all_field_ids, pred_lgb_matrix )
		write_sub ( sub_dirty_oof_test_lgb_fname, test_field_ids, pred_lgb_matrix [ ~train_rows ] )

		print ( "\t\tcat" )
		write_sub ( sub_dirty_oof_all_cat_fname, all_field_ids, pred_cat_matrix )
		write_sub ( sub_dirty_oof_test_cat_fname, test_field_ids, pred_cat_matrix [ ~train_rows ] )

		# Done
		print ( "~" * 100 )
		print ( "Done" )

	# Now combine the dirty predictions
	# using the model weights
	pred_dirty_matrix = opt_lgb_cat ( pred_lgb_matrix, pred_cat_matrix )
	write_sub ( sub_dirty_oof_all_both_fname, all_field_ids, pred_dirty_matrix )
	write_sub ( sub_dirty_oof_test_both_fname, test_field_ids, pred_dirty_matrix [ ~train_rows ] )

	# Report dirty results
	print ( f"Dirty model results" )
	print ( f"OOF" )

	if use_lgb :

		print ( f"\tlgb  : {log_loss ( train_labels, pred_lgb_matrix [ train_rows ] )}" )

	if use_cat :

		print ( f"\tcat  : {log_loss ( train_labels, pred_cat_matrix [ train_rows ] )}" )

	print ( f"\tboth : {log_loss ( train_labels, pred_dirty_matrix [ train_rows ] )}" )

print ( "Done!" )
