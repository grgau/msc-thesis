import math
import sys
import cPickle as pickle
from datetime import datetime
import random
import argparse
import entropy_analysis

def get_ICDs_from_mimic_file(fileName, stayToMap):
  mimicFile = open(fileName, 'r')  # subject_id,stay_id,seq_num,icd_code,icd_version,icd_title
  mimicFile.readline()
  number_of_null_ICD_codes = 0
  for line in mimicFile:
    tokens = line.strip().split(',')
    stay_id = int(tokens[1])
    if (len(tokens[3]) == 0):  # ignore diagnoses where icd_code is null
      number_of_null_ICD_codes += 1
      continue;

    # Remove alphanumeric characters
    tokens[3] = tokens[3].replace('/','')
    tokens[3] = tokens[3].replace('.', '')
    tokens[3] = tokens[3].replace('+', '')
    tokens[3] = tokens[3].replace('"', '')
    
    ICD_code = tokens[3]
    ICD_version = tokens[4]
    filter_ICD_version = False

    if ARGS.icd_version != 'both':
      filter_ICD_version = True

    if filter_ICD_version == False or (filter_ICD_version == True and ICD_version == ARGS.icd_version): 
      if ICD_version == "9":
        ICD_code = ICD_code[:4] # Get only first 4 codes from ICD-9 (Check mimic-iii preprocessing script for more info)
      
      if stay_id in stayToMap:
        stayToMap[stay_id].add(ICD_code)
      else:
        stayToMap[stay_id] = set()              #use set to avoid repetitions
        stayToMap[stay_id].add(ICD_code)

  for stay_id in stayToMap.keys():
    stayToMap[stay_id] = list(stayToMap[stay_id])   #convert to list, as the rest of the codes expects
  mimicFile.close()
  print '-Number of null ICD codes in file ' + fileName + ': ' + str(number_of_null_ICD_codes)

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('admissions_file', type=str, default='edstays.csv', help='The edstays.csv file of Mimic-IV-ED distribution.')
  parser.add_argument('diagnoses_file', type=str, default='', help='The diagnosis.csv file of mimic-iv-ed distribution.')
  parser.add_argument('output_prefix', type=str, default='preprocessing', help='The output file radical name.')
  parser.add_argument('--data_partition', type=str, default='[90,10]', help='Provide an array with two values that sum up 100.')
  parser.add_argument('--icd_version', type=str, default='both', help='Version of ICD code to filter, it can be 9, 10 or both')
  argsTemp = parser.parse_args()
  return argsTemp

if __name__ == '__main__':
  ARGS = parse_arguments()
  partitions = [int(strDim) for strDim in ARGS.data_partition[1:-1].split(',')]
  Ordered_internalCodesMap = {}
  
  #one line of the admissions file contains one admission stay_id of one subject_id at a given time admittime
  print 'Building Maps: stay_id to intime and duration; and Map: subject_id to set of all its stay_ids'
  subjectTOstays_Map = {}
  stayTOintime_Map = {}
  stayTOduration_Map = {}
  stayTOinterval_Map = {}
  mimic_ADMISSIONS_csv = open(ARGS.admissions_file, 'r')
  # subject_id,hadm_id,stay_id,intime,outtime (Here we use stay_id since hadm_id is attribute to relate MIMIC-IV with MIMIC-IV-ED datasets)
  mimic_ADMISSIONS_csv.readline()

  initial_number_of_admissions = 0
  previous_subject = 0
  previous_admission = 0

  for line in mimic_ADMISSIONS_csv:
    initial_number_of_admissions += 1
    tokens = line.strip().split(',')
    subject_id = int(tokens[0])
    stay_id = int(tokens[2])
    intime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
    outtime = datetime.strptime(tokens[4], '%Y-%m-%d %H:%M:%S')

    # stayTOintime_Map(stay_id) -> duration of admission in hours
    temp = outtime - intime
    stayTOduration_Map[stay_id] = temp.days * 24 + temp.seconds / 3600  # duration in hours

    #keep track of the admission amount of time
    #stayTOintime_Map(stay_id) -> time of admission stay_id
    stayTOintime_Map[stay_id] = intime

    #on a subject basis
    if subject_id == previous_subject:
      # keep track of the time since the last admission in days
      temp = intime - stayTOintime_Map[previous_admission]
      stayTOinterval_Map[stay_id] = temp.days + temp.seconds / 3600 / 24  # time since the last admission in days
    else:
      stayTOinterval_Map[stay_id] = 0  # 1st interval since the last admission is 0

    previous_admission = stay_id
    previous_subject = subject_id

    #subjectTOstays_Map(subject_id) -> set of stays for subject_id
    if subject_id in subjectTOstays_Map: subjectTOstays_Map[subject_id].append(stay_id)
    else: subjectTOstays_Map[subject_id] = [stay_id] #the brackets indicate that it will be a list
  
  mimic_ADMISSIONS_csv.close()
  print '-Initial number of admissions: ' + str(initial_number_of_admissions)
  print '-Initial number of subjects: ' + str(len(subjectTOstays_Map))
  stayToICDsCODEs_Map = {}
  
  if len(ARGS.diagnoses_file) > 0:
    #one line in the diagnoses file contains only one diagnose code (ICD) for one admission stay_id
    print 'Building Map: stay_id to set of ICD codes from DIAGNOSES_ICD'
    get_ICDs_from_mimic_file(ARGS.diagnoses_file, stayToICDsCODEs_Map)

  print '-Number of valid admissions (at least one diagnosis): ' + str(len(stayToICDsCODEs_Map))

  #Cleaning up inconsistencies
  #some tuples in the diagnoses table have ICD empty; we clear the admissions without diagnoses from all the maps
  #this may cause the presence of patients (subject_ids) with 0 admissions stay_id; we clear these guys too
  #We also clean admissions in which admission time < outtime
  number_of_admissions_without_diagnosis = 0
  number_of_subjects_without_valid_admissions = 0
  print 'Cleaning up admissions without diagnoses'
  for subject_id, stayList in subjectTOstays_Map.items():   #stayTOintime_Map,subjectTOstays_Map,stay_icds_Map
    stayListCopy = list(stayList)    #copy the list, iterate over the copy, edit the original; otherwise, iteration problems
    for stay_id in stayListCopy:
      if stay_id not in stayToICDsCODEs_Map.keys():  #map stayToICDsCODEs_Map is already valid by creation
        number_of_admissions_without_diagnosis += 1
        del stayTOintime_Map[stay_id]     #delete by key
        del stayTOduration_Map[stay_id]
        del stayTOinterval_Map[stay_id]
        stayList.remove(stay_id)
    if len(stayList) == 0:					      #toss off subject_id without admissions
      number_of_subjects_without_valid_admissions += 1
      del subjectTOstays_Map[subject_id]     #delete by value
  print '-Number of admissions without diagnosis: ' + str(number_of_admissions_without_diagnosis)
  print '-Number of admissions after cleaning: ' + str(len(stayToICDsCODEs_Map))
  print '-Number of subjects without admissions: ' + str(number_of_subjects_without_valid_admissions)
  print '-Number of subjects after cleaning: ' + str(len(subjectTOstays_Map))

  #since the data in the database is not necessarily time-ordered
  #here we sort the admissions (stay_id) according to the admission time (intime)
  #after this, we have a list subjectTOorderedStay_IDS_Map(subject_id) -> admission-time-ordered set of ICD codes
  print 'Building Map: subject_id to admission-ordered (intime, ICDs set) and cleaning one-admission-only patients'
  subjectTOorderedStay_IDS_Map = {}
	#for each admission stay_id of each patient subject_id
  number_of_subjects_with_only_one_visit = 0
  for subject_id, stayList in subjectTOstays_Map.iteritems():
    if len(stayList) < 2:
      number_of_subjects_with_only_one_visit += 1
      continue  #discard subjects with only 2 admissions
    #sorts the stay_ids according to date intime
    #only for the stay_id in the list stayList
    sortedList = sorted([(stayTOintime_Map[stay_id], stayToICDsCODEs_Map[stay_id], stay_id) for stay_id in stayList])
    # each element in subjectTOorderedStay_IDS_Map is a key-value (subject_id, (intime, ICD_List, stay_id))
    subjectTOorderedStay_IDS_Map[subject_id] = sortedList
  print '-Number of discarded subjects with only one admission: ' + str(number_of_subjects_with_only_one_visit)
  print '-Number of subjects after ordering: ' + str(len(subjectTOorderedStay_IDS_Map))

  print 'Converting maps to lists in preparation for dump'
  all_subjectsListOfCODEsList_LIST = []
  #for each subject_id, get its key-value (subject_id, (intime, CODESs_List))
  for subject_id, time_ordered_CODESs_List in subjectTOorderedStay_IDS_Map.iteritems():
    subject_list_of_CODEs_List = []
    #for each admission (intime, CODESs_List) build lists of time and CODEs list
    for admission in time_ordered_CODESs_List:   		#each element in time_ordered_CODESs_List is a tripple (intime, ICD_List, stay_id)
	    #here, admission = [intime, ICD_List, stay_id)
      subject_list_of_CODEs_List.append((admission[1],admission[2]))  #build list of lists of the admissions' CODEs of the current subject_id, stores stay_id together
    #lists of lists, one entry per subject_id
    all_subjectsListOfCODEsList_LIST.append(subject_list_of_CODEs_List)	#build list of list of lists of the admissions' ICDs - one entry per subject_id

  CODES_distributionMAP = entropy_analysis.writeDistributions(ARGS.admissions_file, stayToICDsCODEs_Map, subjectTOstays_Map, all_subjectsListOfCODEsList_LIST)
  for i, key in enumerate(CODES_distributionMAP):
    Ordered_internalCodesMap[key[0]] = i 
  entropy_analysis.computeShannonEntropyDistribution(all_subjectsListOfCODEsList_LIST, CODES_distributionMAP, ARGS.admissions_file)
	
  #Randomize the order of the patients at the first dimension
  random.shuffle(all_subjectsListOfCODEsList_LIST)

  duration_of_admissionsListOfLists = []  #list of lists of duration of admissions, one list for each patient (subjet_id)
  interval_since_last_admissionListOfLists = []
  new_all_subjectsListOfCODEsList_LIST = []
  final_number_of_admissions = 0
  #Here we convert the database codes to internal sequential codes
  #we use the same for to build lists of interval, duration and department
  print 'Converting database ids to sequential integer ids'
  for subject_list_of_CODEs_List in all_subjectsListOfCODEsList_LIST:
    new_subject_list_of_CODEs_List = []
    duration_of_admissionsList = []
    interval_since_last_admissionList = []
    for CODEs_List in subject_list_of_CODEs_List:
      final_number_of_admissions += 1
      new_CODEs_List = []
      stay_id = CODEs_List[1]
      durationTemp = stayTOduration_Map[stay_id]
      intervalTemp = stayTOinterval_Map[stay_id]
      #we bypass admissions with 0 or negative durations
      if durationTemp <= 0 or intervalTemp < 0:
        continue

      duration_of_admissionsList.append(durationTemp)
      interval_since_last_admissionList.append(intervalTemp)

      for CODE in CODEs_List[0]:
        new_CODEs_List.append(Ordered_internalCodesMap[CODE])   #newVisit is the CODEs_List, but with the new sequential ids
      new_subject_list_of_CODEs_List.append(new_CODEs_List)		#new_subject_list_of_CODEs_List is the subject_list_of_CODEs_List, but with the id given by its frequency

    #when we bypass admissions with 0 or negative durations, we might create patients with only one admission, which we also bypass
    if len(new_subject_list_of_CODEs_List) > 1:
      duration_of_admissionsListOfLists.append(duration_of_admissionsList)
      interval_since_last_admissionListOfLists.append(interval_since_last_admissionList)
      new_all_subjectsListOfCODEsList_LIST.append(new_subject_list_of_CODEs_List)	#new_all_subjectsListOfCODEsList_LIST is the all_subjectsListOfCODEsList_LIST, but with the new sequential ids

  print ''
  nCodes = len(Ordered_internalCodesMap)
  print '-Number of actually used DIAGNOSES codes: '+ str(nCodes)

  print '-Final number of subjects: ' + str(len(new_all_subjectsListOfCODEsList_LIST))
  print '-Final number of admissions: ' + str(final_number_of_admissions)

  #Partitioning data
  if (len(partitions) >= 1):
    total_patients_dumped = 0
    print 'Writing ' + str(partitions[0]) + '% of the patients read from file ' + ARGS.admissions_file
    index_of_last_patient_to_dump = int(math.ceil(len(new_all_subjectsListOfCODEsList_LIST)*int(partitions[0])/100))
    pickle.dump(new_all_subjectsListOfCODEsList_LIST[0:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.train', 'wb'), -1)
    pickle.dump(duration_of_admissionsListOfLists[0:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.DURATION.train', 'wb'), -1)
    pickle.dump(interval_since_last_admissionListOfLists[0:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.INTERVAL.train', 'wb'), -1)
    print '   Patients from 0 to ' + str(index_of_last_patient_to_dump)
    print '   Success, file: ' + ARGS.output_prefix + '_' + str(nCodes) + '.train created'
    total_patients_dumped += index_of_last_patient_to_dump

    if (len(partitions) >= 2):
      print 'Writing ' + str(partitions[1]) + '% of the patients read from file ' + ARGS.admissions_file
      index_of_first_patient_to_dump = index_of_last_patient_to_dump
      index_of_last_patient_to_dump = index_of_first_patient_to_dump + int(math.ceil(len(new_all_subjectsListOfCODEsList_LIST)*int(partitions[1])/100))
      pickle.dump(new_all_subjectsListOfCODEsList_LIST[index_of_first_patient_to_dump:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.test', 'wb'), -1)
      pickle.dump(duration_of_admissionsListOfLists[index_of_first_patient_to_dump:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.DURATION.test', 'wb'), -1)
      pickle.dump(interval_since_last_admissionListOfLists[index_of_first_patient_to_dump:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.INTERVAL.test', 'wb'), -1)
      print '   Patients from ' + str(index_of_first_patient_to_dump) + ' to ' + str(index_of_last_patient_to_dump)
      print '   Success, file: ' + ARGS.output_prefix + '_' + str(nCodes) + '.test created'
      total_patients_dumped += index_of_last_patient_to_dump - index_of_first_patient_to_dump

      if (len(partitions) >= 3):
        print 'Writing ' + str(partitions[2]) + '% of the patients read from file ' + ARGS.admissions_file
        index_of_first_patient_to_dump = index_of_last_patient_to_dump
        pickle.dump(new_all_subjectsListOfCODEsList_LIST[index_of_first_patient_to_dump:],open(ARGS.output_prefix + '_' + str(nCodes) + '.valid', 'wb'), -1)
        pickle.dump(duration_of_admissionsListOfLists[index_of_first_patient_to_dump:], open(ARGS.output_prefix + '_' + str(nCodes) + '.DURATION.valid', 'wb'), -1)
        pickle.dump(interval_since_last_admissionListOfLists[index_of_first_patient_to_dump:], open(ARGS.output_prefix + '_' + str(nCodes) + '.INTERVAL.valid', 'wb'), -1)
        print '   Patients from ' + str(index_of_first_patient_to_dump) + ' to the end of the file'
        print '   Success, file: ' + ARGS.output_prefix + '_' + str(nCodes) + '.valid created'
        total_patients_dumped += len(new_all_subjectsListOfCODEsList_LIST) - total_patients_dumped
        print 'Total of dumped patients: ' + str(total_patients_dumped) + ' out of ' + str(len(new_all_subjectsListOfCODEsList_LIST))
  else:
    print 'Error, please provide data partition scheme. E.g, [80,10,10], for 80\% train, 10\% test, and 10\% validation.'
