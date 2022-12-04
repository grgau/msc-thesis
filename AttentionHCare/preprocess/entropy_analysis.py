#################################################################################################
# author: junio@usp.br - Jose F Rodrigues Jr
#################################################################################################
import math

def writeDistributions(inputFile, hadmToCodes_Map = None, subjectTOhadms_Map = None, all_subjects_list_of_CODEs_List = None):
	if inputFile[-4:-3] == '.':
		inputFile = inputFile[0:-3]
	distFileNamePrefix = inputFile + 'DISTRIBUTION.'
	#----------------
	if hadmToCodes_Map is not None:
		print 'Distribution of the number of codes per admission'
		maxNumberOfCodesPerVisit = 0
		for hadm_id in hadmToCodes_Map.keys():
			if len(hadmToCodes_Map[hadm_id]) > maxNumberOfCodesPerVisit:
				maxNumberOfCodesPerVisit = len(hadmToCodes_Map[hadm_id])
		nCodesPerAdmissionDistributionLIST = [0] * (maxNumberOfCodesPerVisit+1)
		for hadm_id in hadmToCodes_Map.keys():
			nCodesPerAdmissionDistributionLIST[len(hadmToCodes_Map[hadm_id])] += 1

		totalCount = 0
		for i in range(0, len(nCodesPerAdmissionDistributionLIST)):
			totalCount += nCodesPerAdmissionDistributionLIST[i]
		accumulated = 0

		file = open(distFileNamePrefix + 'nCodesPerAdmission.csv', 'w')
		file.write('nCodes;frequency;percentage;accumulated' + '\n')
		for i in range(0,len(nCodesPerAdmissionDistributionLIST)):
			if nCodesPerAdmissionDistributionLIST[i] > 0:
				percentage = nCodesPerAdmissionDistributionLIST[i]/float(totalCount)
				accumulated += percentage
				file.write(str(i) + "; " + str(nCodesPerAdmissionDistributionLIST[i]))
				file.write(";" + "%.2f"%percentage)
				file.write(";" + "%.2f" % accumulated + '\n')
		file.close()
	#----------------
	if subjectTOhadms_Map is not None:
		print 'Distribution of the number of admissions per patient'
		maxNumberOfVisits = 0
		for subject_id, hadmList in subjectTOhadms_Map.iteritems():
			if len(hadmList) > maxNumberOfVisits:
				maxNumberOfVisits = len(hadmList)

		nAdmissionsDistributionLIST = [0]*(maxNumberOfVisits+1)
		for subject_id, hadmList in subjectTOhadms_Map.iteritems():
			nAdmissionsDistributionLIST[len(hadmList)] += 1

		totalCount = 0
		for i in range(0, len(nAdmissionsDistributionLIST)):
			totalCount += nAdmissionsDistributionLIST[i]

		accumulated = 0
		file = open(distFileNamePrefix + 'nAdmissionsPerPatient.csv', 'w')
		file.write('nAdmissions;frequency;percentage;accumulated' + '\n')
		for i in range(0,len(nAdmissionsDistributionLIST)):
			if(nAdmissionsDistributionLIST[i] > 0):
				percentage = nAdmissionsDistributionLIST[i]/float(totalCount)
				accumulated += percentage
				file.write(str(i) + "; " + str(nAdmissionsDistributionLIST[i]))
				file.write(";" + "%.2f"%percentage)
				file.write(";" + "%.2f" % accumulated + '\n')
		file.close()
	#----------------
	if all_subjects_list_of_CODEs_List is not None:
		#distribution of number of times each code is used
		print 'Distribution of the number of times each code is used'
		CODES_distributionMAP = {}	#Here we build the distribution of codes and order the inner codes of the network according to this distribution
		#As a result, the most common code will become 0-indexed in its respective tensor dimension
		for subject_list_of_CODEs_List in all_subjects_list_of_CODEs_List:
			for CODEs_List in subject_list_of_CODEs_List:
				for CODE in CODEs_List[0]:
					if CODE in CODES_distributionMAP:
						CODES_distributionMAP[CODE] += 1
					else:
						CODES_distributionMAP[CODE] = 1
		CODES_distributionMAP = sorted(CODES_distributionMAP.items(), key=lambda x: x[1],reverse=True)

		totalCount = 0
		for CODE, value in CODES_distributionMAP:
			totalCount += value

		accumulated = 0
		file = open(distFileNamePrefix + 'nUsesPerCode.csv', 'w')
		file.write('code;frequency;percentage;accumulated' + '\n')
		for CODE, value in CODES_distributionMAP:
			percentage = value / float(totalCount)
			accumulated += percentage
			file.write(str(CODE) + '; ' + str(value))
			file.write(";" + "%.2f" % percentage)
			file.write(";" + "%.2f" % accumulated + '\n')
		file.close()

		print 'Distribution files written at ' + distFileNamePrefix
	return CODES_distributionMAP
	#----------------
#computes the entropy of every sequence in .... and draws the corresponding distribution of entropies
def computeShannonEntropyDistribution(all_subjectsListOfCODEsList_LIST, codesFrequencyMAP, outputFile = 'shannon'):
	if outputFile[-4:-3] == '.':
		outputFile = outputFile[0:-3] + 'shannon'
	outputFile = outputFile + 'DISTRIBUTION.csv'
	#computer the simple probability of each code
	totalSum = 0
	probabilityIndexes_MAP = {}
	for code,frequency in codesFrequencyMAP:
		totalSum += frequency
	for code,frequency in codesFrequencyMAP:
		probabilityIndexes_MAP[code] = frequency/float(totalSum)

	shannonEntropy = 0
	shannonEntropy_LIST = []
	maxEntropy = 0
	minEntropy = 1000
	for patient in all_subjectsListOfCODEsList_LIST:
		for admission in patient:
			shannonEntropy = 0
			for code in admission[0]:
				shannonEntropy += -probabilityIndexes_MAP[code]*math.log(probabilityIndexes_MAP[code])
			shannonEntropy_LIST.append(shannonEntropy)
			if shannonEntropy > maxEntropy: maxEntropy = shannonEntropy
			if shannonEntropy < minEntropy: minEntropy = shannonEntropy

	numberOfBins = 30#int(math.ceil(math.sqrt(len(shannonEntropy_LIST))))
	binWidth = 0.1#(maxEntropy-minEntropy)/numberOfBins

	bin_MAP = {}
	for i in range(0, numberOfBins):
		bin_MAP[i] = 0
	totalCounts = 0
	for i in range(0,numberOfBins):
		binLowerLimit = i*binWidth
		binUpperLimit =	(i+1)*binWidth
		for shannonEntropy in shannonEntropy_LIST:
			if shannonEntropy >= binLowerLimit and shannonEntropy < binUpperLimit:
				bin_MAP[i] = bin_MAP[i] + 1
				totalCounts += 1

	file = open(outputFile, 'w')
	file.write('bin;frequency;percentage;accumulated' + '\n')
	accumulated = 0
	for bin in bin_MAP:
		accumulated += bin_MAP[bin]/float(totalCounts)
		file.write("%.2f"%(bin*binWidth)+"-"+"%.2f"%((bin+1)*binWidth))
		file.write('; ' + str(bin_MAP[bin]))
		file.write('; ' + "%.2f"%(bin_MAP[bin]/float(totalCounts)))
		file.write(";" + "%.2f"%accumulated +'\n')
	file.close()
	print 'Shannon distribution file written at ' + outputFile
