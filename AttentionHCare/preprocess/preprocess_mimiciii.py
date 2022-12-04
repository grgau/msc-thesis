#################################################################################################
# author: junio@usp.br - Jose F Rodrigues Jr
# This script extracts data from Mimic-III files ADMISSIONS.csv and DIAGNOSES_icd.csv preparing it for LIG-Doctor.py
# The output is in format cPickled: a list of patients, each one with a list of admissions, each admission with a list of diagnosis codes
# Minimum input: files ADMISSIONS.csv, DIAGNOSES_ICD.csv
# Example: python ./preprocess_mimiciii.py ./data/ADMISSIONS.csv ./data/DIAGNOSES_ICD.csv ./data/preprocess/outputName
# Note: the time-related stuff assumes that the ADMISSIONS.csv file was exported from postgres with command
#\copy (SELECT * FROM ADMISSIONS ORDER BY SUBJECT_ID, ADMITTIME ASC) TO 'ADMISSIONS.csv' WITH (FORMAT CSV,HEADER TRUE)
# Note2: this script does a lot of things, some of them experimental - so, do not worry about everything unless you are willing to extend it
#################################################################################################

import math
import sys
import cPickle as pickle
from datetime import datetime
import random
import argparse
import entropy_analysis

global ARGS

#given a map of (hadm_id, set of diagnoses icd9 codes), convert the map to (hadm_id, CCS codes)
def map_ICD9_to_CCS(map):
	icd9TOCCS_Map = pickle.load(open(sys.path[0]+'/icd9_to_css_dictionary','rb'))
	procCODEstoInternalID_map = {}
	set_of_used_codes = set()
	for (hadm_id, ICD9s_List) in map.items():
		for ICD9 in ICD9s_List:
			while (len(ICD9) < 6): ICD9 += ' '  #pad right white spaces because the CCS mapping uses this pattern
			try:
				CCS_code = icd9TOCCS_Map[ICD9]
				if hadm_id in procCODEstoInternalID_map:
					procCODEstoInternalID_map[hadm_id].append(CCS_code)
				else:
					procCODEstoInternalID_map[hadm_id] = [CCS_code]
				set_of_used_codes.add(ICD9)
			except KeyError:
				print str(sys.exc_info()[0]) + '  ' + str(ICD9) + ". ICD9 code not found, please verify your ICD9 to CCS mapping before proceeding."
	print '-Total number (complete set) of ICD9 codes (diag + proc): ' + str(len(set(icd9TOCCS_Map.keys())))
	print '-Total number (complete set) of CCS codes (diag + proc): ' + str(len(set(icd9TOCCS_Map.values())))
	print '-Total number of ICD9 codes actually used: ' + str(len(set_of_used_codes))

	return procCODEstoInternalID_map

def get_ICD9s_from_mimic_file(fileName, hadmToMap):
	mimicFile = open(fileName, 'r')  # row_id,subject_id,hadm_id,seq_num,ICD9_code
	mimicFile.readline()
	number_of_null_ICD9_codes = 0
	for line in mimicFile:			 #   0  ,     1    ,    2   ,   3  ,    4
		tokens = line.strip().split(',')
		hadm_id = int(tokens[2])
		if (len(tokens[4]) == 0):  # ignore diagnoses where ICD9_code is null
			number_of_null_ICD9_codes += 1
			continue;

		ICD9_code = tokens[4]
		if ICD9_code.find("\"") != -1:
			ICD9_code = ICD9_code[1:-1]  # toss off quotes and proceed
		# since diagnosis and procedure ICD9 codes have intersections, a prefix is necessary for disambiguation
		if fileName == ARGS.diagnoses_file:
			ICD9_code = 'D' + ICD9_code
		else:
			ICD9_code = 'P' + ICD9_code
		# To understand the line below, check https://mimic.physionet.org/mimictables/diagnoses_icd/
		# "The code field for the ICD-9-CM Principal and Other Diagnosis Codes is six characters in length (not really!),
		# with the decimal point implied between the third and fourth digit for all diagnosis codes other than the V codes.
		# The decimal is implied for V codes between the second and third digit."
		# Actually, if you look at the codes (https://raw.githubusercontent.com/drobbins/ICD9/master/ICD9.txt), simply take the three first characters
		if not ARGS.map_ICD9_to_CCS:
			ICD9_code = ICD9_code[:4]  # No CCS mapping, get the first alphanumeric four letters only

		if hadm_id in hadmToMap:
			hadmToMap[hadm_id].add(ICD9_code)
		else:
			hadmToMap[hadm_id] = set()              #use set to avoid repetitions
			hadmToMap[hadm_id].add(ICD9_code)
	for hadm_id in hadmToMap.keys():
		hadmToMap[hadm_id] = list(hadmToMap[hadm_id])   #convert to list, as the rest of the codes expects
	mimicFile.close()
	print '-Number of null ICD9 codes in file ' + fileName + ': ' + str(number_of_null_ICD9_codes)

def convert_type_to_float(type):
	#very specific to Mimic-III ADMISSIONS.csv
	code = 0
	if type == 'NEWBORN': code = 0
	elif type == 'ELECTIVE': code = 1
	elif type == 'EMERGENCY': code = 2
	elif type == 'URGENT': code = 3
	else: print 'ERROR in admission type value'

	return code

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('admissions_file', type=str, default='ADMISSIONS.csv', help='The ADMISSIONS.csv file of Mimic-iii distribution.')
	parser.add_argument('diagnoses_file', type=str, default='', help='The DIAGNOSES_ICD.csv file of mimic-iii distribution.')
	parser.add_argument('output_prefix', type=str, default='preprocessing', help='The output file radical name.')
	parser.add_argument('--procedures_file', type=str, default='', help='The optional PROCEDURES_ICD.csv file of mimic-iii distribution - for processing using procedures codes.')
	parser.add_argument('--map_ICD9_to_CCS', type=int, default=0, choices=[0,1], help='True/False [0/1] to ap ICD9 codes to CCS codes (better accuracy, less granularity); refer to https://www.hcup-us.ahrq.gov/toolssoftware/CCS/CCS.jsp.')
	parser.add_argument('--data_partition', type=str, default='[90,10]', help='Provide an array with two values that sum up 100.')
	argsTemp = parser.parse_args()
	return argsTemp

if __name__ == '__main__':
	global ARGS
	ARGS = parse_arguments()
	partitions = [int(strDim) for strDim in ARGS.data_partition[1:-1].split(',')]
	CCS_ordered_internalCodesMap = {}

	#one line of the admissions file contains one admission hadm_id of one subject_id at a given time admittime
	print 'Building Maps: hadm_id to admtttime, duration, and type; and Map: subject_id to set of all its hadm_ids'
	subjectTOhadms_Map = {}
	hadmTOadmttime_Map = {}					   					#   0  ,     1    ,    2  ,     3   ,    4
	hadmTOduration_Map = {}
	hadmTOinterval_Map = {}
	hadmTOadmtype_Map = {}
	mimic_ADMISSIONS_csv = open(ARGS.admissions_file, 'r')
	# row_id,subject_id,hadm_id,admittime,dischtime,deathtime,admission_type,admission_location,discharge_location,insurance,language,religion,marital_status,ethnicity,
	mimic_ADMISSIONS_csv.readline()

	initial_number_of_admissions = 0
	previous_subject = 0
	previous_admission = 0
	for line in mimic_ADMISSIONS_csv:
		initial_number_of_admissions += 1
		tokens = line.strip().split(',')
		subject_id = int(tokens[1])
		hadm_id = int(tokens[2])
		admittime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
		dischargetime = datetime.strptime(tokens[4], '%Y-%m-%d %H:%M:%S')
		admissionType = tokens[6]

		# hadmTOadmttime_Map(hadm_id) -> duration of admission in hours
		temp = dischargetime - admittime
		hadmTOduration_Map[hadm_id] = temp.days * 24 + temp.seconds / 3600  # duration in hours

		#the time-related stuff that follows, assumes that the ADMISSIONS.csv file
		#was exported from postgres with command
		#\copy (SELECT * FROM ADMISSIONS ORDER BY SUBJECT_ID, ADMITTIME ASC) TO '/j/dat/mimic-original/ADMISSIONS.csv' WITH (FORMAT CSV,HEADER TRUE)

		#keep track of the admission amount of time
		#hadmTOadmttime_Map(hadm_id) -> time of admission hadm_id
		hadmTOadmttime_Map[hadm_id] = admittime

		#on a subject basis
		if subject_id == previous_subject:
			# keep track of the time since the last admission in days
			temp = admittime - hadmTOadmttime_Map[previous_admission]
			hadmTOinterval_Map[hadm_id] = temp.days + temp.seconds / 3600 / 24  # time since the last admission in days
		else:
			hadmTOinterval_Map[hadm_id] = 0  # 1st interval since the last admission is 0

		previous_admission = hadm_id
		previous_subject = subject_id

		#register type of admission
		hadmTOadmtype_Map[hadm_id] = [convert_type_to_float(admissionType)]

		#subjectTOhadms_Map(subject_id) -> set of hadms for subject_id
		if subject_id in subjectTOhadms_Map: subjectTOhadms_Map[subject_id].append(hadm_id)
		else: subjectTOhadms_Map[subject_id] = [hadm_id] #the brackets indicate that it will be a list
	mimic_ADMISSIONS_csv.close()
	print '-Initial number of admissions: ' + str(initial_number_of_admissions)
	print '-Initial number of subjects: ' + str(len(subjectTOhadms_Map))
	hadmToICD9CODEs_Map = {}
	hadmToICD9ProcCODEs_Map = {}

	if len(ARGS.diagnoses_file) > 0:
		#one line in the diagnoses file contains only one diagnose code (ICD9) for one admission hadm_id
		print 'Building Map: hadm_id to set of ICD9 codes from DIAGNOSES_ICD'
		get_ICD9s_from_mimic_file(ARGS.diagnoses_file, hadmToICD9CODEs_Map)
	if len(ARGS.procedures_file) > 0:
		print 'Building Map: hadm_id to set of ICD9 codes from PROCEDURES_ICD'
		get_ICD9s_from_mimic_file(ARGS.procedures_file, hadmToICD9ProcCODEs_Map)

	print '-Number of valid admissions (at least one diagnosis): ' + str(len(hadmToICD9CODEs_Map))

	#Here we make sure we have only admissions with procedures AND diagnosis
	if len(ARGS.procedures_file) > 0:
		print '-Number of procedures codes before cleaning: ' + str(len(hadmToICD9ProcCODEs_Map))
		for hadm_id in hadmToICD9CODEs_Map.keys():
			if hadm_id not in hadmToICD9ProcCODEs_Map.keys():
				del hadmToICD9CODEs_Map[hadm_id]
		for hadm_id in hadmToICD9ProcCODEs_Map.keys():
			if hadm_id not in hadmToICD9CODEs_Map.keys():
				del hadmToICD9ProcCODEs_Map[hadm_id]
		print '-Number of procedures after cleaning: ' + str(len(hadmToICD9ProcCODEs_Map))

	#Cleaning up inconsistencies
	#some tuples in the diagnoses table have ICD9 empty; we clear the admissions without diagnoses from all the maps
	#this may cause the presence of patients (subject_ids) with 0 admissions hadm_id; we clear these guys too
	#We also clean admissions in which admission time < discharge time - there are 89 records like that in the original dataset
	number_of_admissions_without_diagnosis = 0
	number_of_subjects_without_valid_admissions = 0
	print 'Cleaning up admissions without diagnoses'
	for subject_id, hadmList in subjectTOhadms_Map.items():   #hadmTOadmttime_Map,subjectTOhadms_Map,hadm_cid9s_Map
		hadmListCopy = list(hadmList)    #copy the list, iterate over the copy, edit the original; otherwise, iteration problems
		for hadm_id in hadmListCopy:
			if hadm_id not in hadmToICD9CODEs_Map.keys():  #map hadmToICD9CODEs_Map is already valid by creation
				number_of_admissions_without_diagnosis += 1
				del hadmTOadmttime_Map[hadm_id]     #delete by key
				del hadmTOduration_Map[hadm_id]
				del hadmTOinterval_Map[hadm_id]
				del hadmTOadmtype_Map[hadm_id]
				hadmList.remove(hadm_id)
		if len(hadmList) == 0:					      #toss off subject_id without admissions
			number_of_subjects_without_valid_admissions += 1
			del subjectTOhadms_Map[subject_id]     #delete by value
	print '-Number of admissions without diagnosis: ' + str(number_of_admissions_without_diagnosis)
	print '-Number of admissions after cleaning: ' + str(len(hadmToICD9CODEs_Map))
	print '-Number of subjects without admissions: ' + str(number_of_subjects_without_valid_admissions)
	print '-Number of subjects after cleaning: ' + str(len(subjectTOhadms_Map))

	if ARGS.map_ICD9_to_CCS:
		print 'Mapping ICD9 codes to CCS codes'
		hadmToICD9CODEs_Map = map_ICD9_to_CCS(hadmToICD9CODEs_Map)
		if len(ARGS.procedures_file) > 0:
			hadmToICD9ProcCODEs_Map = map_ICD9_to_CCS(hadmToICD9ProcCODEs_Map)

	#since the data in the database is not necessarily time-ordered
	#here we sort the admissions (hadm_id) according to the admission time (admittime)
	#after this, we have a list subjectTOorderedHADM_IDS_Map(subject_id) -> admission-time-ordered set of ICD9 codes
	print 'Building Map: subject_id to admission-ordered (admittime, ICD9s set) and cleaning one-admission-only patients'
	subjectTOorderedHADM_IDS_Map = {}
	subjectTOProcHADM_IDs_Map = {}
	#for each admission hadm_id of each patient subject_id
	number_of_subjects_with_only_one_visit = 0
	for subject_id, hadmList in subjectTOhadms_Map.iteritems():
		if len(hadmList) < 2:
			number_of_subjects_with_only_one_visit += 1
			continue  #discard subjects with only 2 admissions
		#sorts the hadm_ids according to date admttime
		#only for the hadm_id in the list hadmList
		sortedList = sorted([(hadmTOadmttime_Map[hadm_id], hadmToICD9CODEs_Map[hadm_id], hadm_id) for hadm_id in hadmList])
		# each element in subjectTOorderedHADM_IDS_Map is a key-value (subject_id, (admittime, ICD9_List, hadm_id))
		subjectTOorderedHADM_IDS_Map[subject_id] = sortedList
	print '-Number of discarded subjects with only one admission: ' + str(number_of_subjects_with_only_one_visit)
	print '-Number of subjects after ordering: ' + str(len(subjectTOorderedHADM_IDS_Map))

	print 'Converting maps to lists in preparation for dump'
	all_subjectsListOfCODEsList_LIST = []
	#for each subject_id, get its key-value (subject_id, (admittime, CODESs_List))
	for subject_id, time_ordered_CODESs_List in subjectTOorderedHADM_IDS_Map.iteritems():
		subject_list_of_CODEs_List = []
		#for each admission (admittime, CODESs_List) build lists of time and CODEs list
		for admission in time_ordered_CODESs_List:   		#each element in time_ordered_CODESs_List is a tripple (admittime, ICD9_List, hadm_id)
			#here, admission = [admittime, ICD9_List, hadm_id)
			subject_list_of_CODEs_List.append((admission[1],admission[2]))  #build list of lists of the admissions' CODEs of the current subject_id, stores hadm_id together
		#lists of lists, one entry per subject_id
		all_subjectsListOfCODEsList_LIST.append(subject_list_of_CODEs_List)	#build list of list of lists of the admissions' ICD9s - one entry per subject_id

	CODES_distributionMAP = entropy_analysis.writeDistributions(ARGS.admissions_file, hadmToICD9CODEs_Map, subjectTOhadms_Map, all_subjectsListOfCODEsList_LIST)
	for i, key in enumerate(CODES_distributionMAP):
		CCS_ordered_internalCodesMap[key[0]] = i
	entropy_analysis.computeShannonEntropyDistribution(all_subjectsListOfCODEsList_LIST, CODES_distributionMAP, ARGS.admissions_file)
	
	#print distribution of CCS codes
	if ARGS.map_ICD9_to_CCS:
		CCS_to_descriptionMap = pickle.load(open(sys.path[0] + '/ccs_to_description_dictionary', 'rb'))
		for CODE, value in CODES_distributionMAP:
			print str(CODE) +': ' + str(value) + ' - ' + CCS_to_descriptionMap[CODE]

	#Randomize the order of the patients at the first dimension
	random.shuffle(all_subjectsListOfCODEsList_LIST)

	duration_of_admissionsListOfLists = []  #list of lists of duration of admissions, one list for each patient (subjet_id)
	interval_since_last_admissionListOfLists = []
	type_of_admissionsListOfLists = []
	new_all_subjectsListOfCODEsList_LIST = []
	new_all_subjects_list_of_ProcCodes_List = []
	final_number_of_admissions = 0
	#Here we convert the database codes to internal sequential codes
	#we use the same for to build lists of interval, duration and type
	print 'Converting database ids to sequential integer ids'
	procCODEstoInternalID_map = {}
	for subject_list_of_CODEs_List in all_subjectsListOfCODEsList_LIST:
		new_subject_list_of_CODEs_List = []
		new_subject_list_of_ProcCODEs_List = []
		duration_of_admissionsList = []
		interval_since_last_admissionList = []
		type_of_admissionsList = []
		for CODEs_List in subject_list_of_CODEs_List:
			final_number_of_admissions += 1
			new_CODEs_List = []
			new_ProcCodes_List = []
			hadm_id = CODEs_List[1]
			durationTemp = hadmTOduration_Map[hadm_id]
			intervalTemp = hadmTOinterval_Map[hadm_id]
			#we bypass admissions with 0 or negative durations
			if durationTemp <= 0 or intervalTemp < 0:
				continue

			duration_of_admissionsList.append(durationTemp)
			interval_since_last_admissionList.append(intervalTemp)
			type_of_admissionsList.append(hadmTOadmtype_Map[hadm_id])

			for CODE in CODEs_List[0]:
				new_CODEs_List.append(CCS_ordered_internalCodesMap[CODE])   #newVisit is the CODEs_List, but with the new sequential ids
			new_subject_list_of_CODEs_List.append(new_CODEs_List)		#new_subject_list_of_CODEs_List is the subject_list_of_CODEs_List, but with the id given by its frequency

			if len(ARGS.procedures_file) > 0:
				for procCODE in hadmToICD9ProcCODEs_Map[CODEs_List[1]]:
					if procCODE not in procCODEstoInternalID_map:
						procCODEstoInternalID_map[procCODE] = len(procCODEstoInternalID_map)
					new_ProcCodes_List.append(procCODEstoInternalID_map[procCODE])
				new_subject_list_of_ProcCODEs_List.append(new_ProcCodes_List)

		#when we bypass admissions with 0 or negative durations, we might create patients with only one admission, which we also bypass
		if len(new_subject_list_of_CODEs_List) > 1:
			duration_of_admissionsListOfLists.append(duration_of_admissionsList)
			interval_since_last_admissionListOfLists.append(interval_since_last_admissionList)
			type_of_admissionsListOfLists.append(type_of_admissionsList)
			new_all_subjectsListOfCODEsList_LIST.append(new_subject_list_of_CODEs_List)	#new_all_subjectsListOfCODEsList_LIST is the all_subjectsListOfCODEsList_LIST, but with the new sequential ids
			if len(ARGS.procedures_file) > 0:
				new_all_subjects_list_of_ProcCodes_List.append(new_subject_list_of_ProcCODEs_List)

	print ''
	nCodes = len(CCS_ordered_internalCodesMap)
	print '-Number of actually used DIAGNOSES codes: '+ str(nCodes)
	if len(ARGS.procedures_file) > 0:
		nProcCodes = len(procCODEstoInternalID_map)
		print '-Numer of actually used PROCEDURE codes: ' + str(nProcCodes)

	print '-Final number of subjects: ' + str(len(new_all_subjectsListOfCODEsList_LIST))
	print '-Final number of admissions: ' + str(final_number_of_admissions)
	#Partitioning data
	if (len(partitions) >= 1):
		total_patients_dumped = 0;
		print 'Writing ' + str(partitions[0]) + '% of the patients read from file ' + ARGS.admissions_file
		index_of_last_patient_to_dump = int(math.ceil(len(new_all_subjectsListOfCODEsList_LIST)*int(partitions[0])/100))
		pickle.dump(new_all_subjectsListOfCODEsList_LIST[0:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.train', 'wb'), -1)
		pickle.dump(duration_of_admissionsListOfLists[0:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.DURATION.train', 'wb'), -1)
		pickle.dump(interval_since_last_admissionListOfLists[0:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.INTERVAL.train', 'wb'), -1)
		pickle.dump(type_of_admissionsListOfLists[0:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.TYPE.train', 'wb'), -1)
		print '   Patients from 0 to ' + str(index_of_last_patient_to_dump)
		print '   Success, file: ' + ARGS.output_prefix + '_' + str(nCodes) + '.train created'
		total_patients_dumped += index_of_last_patient_to_dump
#		if len(ARGS.procedures_file) > 0:
#			pickle.dump(new_all_subjects_list_of_ProcCodes_List[0:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nProcCodes) + '.PROCEDURE.train', 'wb'), -1)

		if (len(partitions) >= 2):
			print 'Writing ' + str(partitions[1]) + '% of the patients read from file ' + ARGS.admissions_file
			index_of_first_patient_to_dump = index_of_last_patient_to_dump
			index_of_last_patient_to_dump = index_of_first_patient_to_dump + int(math.ceil(len(new_all_subjectsListOfCODEsList_LIST)*int(partitions[1])/100))
			pickle.dump(new_all_subjectsListOfCODEsList_LIST[index_of_first_patient_to_dump:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.test', 'wb'), -1)
			pickle.dump(duration_of_admissionsListOfLists[index_of_first_patient_to_dump:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.DURATION.test', 'wb'), -1)
			pickle.dump(interval_since_last_admissionListOfLists[index_of_first_patient_to_dump:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.INTERVAL.test', 'wb'), -1)
			pickle.dump(type_of_admissionsListOfLists[index_of_first_patient_to_dump:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.TYPE.test', 'wb'), -1)
			print '   Patients from ' + str(index_of_first_patient_to_dump) + ' to ' + str(index_of_last_patient_to_dump)
			print '   Success, file: ' + ARGS.output_prefix + '_' + str(nCodes) + '.test created'
			total_patients_dumped += index_of_last_patient_to_dump - index_of_first_patient_to_dump
#			if len(ARGS.procedures_file) > 0:
#				pickle.dump(new_all_subjects_list_of_ProcCodes_List[index_of_first_patient_to_dump:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nProcCodes) + '.PROCEDURE.test', 'wb'), -1)

			if (len(partitions) >= 3):
				print 'Writing ' + str(partitions[2]) + '% of the patients read from file ' + ARGS.admissions_file
				index_of_first_patient_to_dump = index_of_last_patient_to_dump
				pickle.dump(new_all_subjectsListOfCODEsList_LIST[index_of_first_patient_to_dump:],open(ARGS.output_prefix + '_' + str(nCodes) + '.valid', 'wb'), -1)
				pickle.dump(duration_of_admissionsListOfLists[index_of_first_patient_to_dump:], open(ARGS.output_prefix + '_' + str(nCodes) + '.DURATION.valid', 'wb'), -1)
				pickle.dump(interval_since_last_admissionListOfLists[index_of_first_patient_to_dump:], open(ARGS.output_prefix + '_' + str(nCodes) + '.INTERVAL.valid', 'wb'), -1)
				pickle.dump(type_of_admissionsListOfLists[index_of_first_patient_to_dump:], open(ARGS.output_prefix + '_' + str(nCodes) + '.TYPE.valid', 'wb'), -1)
				print '   Patients from ' + str(index_of_first_patient_to_dump) + ' to the end of the file'
				print '   Success, file: ' + ARGS.output_prefix + '_' + str(nCodes) + '.valid created'
				total_patients_dumped += len(new_all_subjectsListOfCODEsList_LIST) - total_patients_dumped
				print 'Total of dumped patients: ' + str(total_patients_dumped) + ' out of ' + str(len(new_all_subjectsListOfCODEsList_LIST))
#				if len(ARGS.procedures_file) > 0:
#					pickle.dump(new_all_subjects_list_of_ProcCodes_List[index_of_first_patient_to_dump:], open(ARGS.output_prefix + '_' + str(nProcCodes) + '.PROCEDURE.valid', 'wb'), -1)
	else:
		print 'Error, please provide data partition scheme. E.g, [80,10,10], for 80\% train, 10\% test, and 10\% validation.'



