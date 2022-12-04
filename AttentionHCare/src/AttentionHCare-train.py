import pickle
import argparse
import shutil
import os
import random

import tensorflow as tf
import numpy as np
from itertools import zip_longest
# from matplotlib import pyplot as plt

global ARGS

def getNumberOfCodes(sets):
  highestCode = 0
  for set in sets:
    for pat in set:
      for adm in pat:
        for code in adm:
          if code > highestCode:
            highestCode = code
  return (highestCode + 1)


def prepareHotVectors(train_tensor, labels_tensor):
  nVisitsOfEachPatient_List = np.array([len(seq) for seq in train_tensor]) - 1
  numberOfPatients = len(train_tensor)
  maxNumberOfAdmissions = np.max(nVisitsOfEachPatient_List)

  x_hotvectors_tensorf = np.zeros((maxNumberOfAdmissions, numberOfPatients, ARGS.numberOfInputCodes)).astype(np.float64)
  y_hotvectors_tensor = np.zeros((maxNumberOfAdmissions, numberOfPatients, ARGS.numberOfInputCodes)).astype(np.float64)
  mask = np.zeros((maxNumberOfAdmissions, numberOfPatients)).astype(np.float64)

  for idx, (train_patient_matrix,label_patient_matrix) in enumerate(zip(train_tensor, labels_tensor)):
    for i_th_visit, visit_line in enumerate(train_patient_matrix[:-1]): #ignores the last admission, which is not part of the training
      for code in visit_line:
        x_hotvectors_tensorf[i_th_visit, idx, code] = 1
    for i_th_visit, visit_line in enumerate(label_patient_matrix[1:]):  #label_matrix[1:] = all but the first admission slice, not used to evaluate (this is the answer)
      for code in visit_line:
        y_hotvectors_tensor[i_th_visit, idx, code] = 1
    mask[:nVisitsOfEachPatient_List[idx], idx] = 1.

  nVisitsOfEachPatient_List = np.array(nVisitsOfEachPatient_List, dtype=np.int32)
  return x_hotvectors_tensorf, y_hotvectors_tensor, mask, nVisitsOfEachPatient_List


def load_data():
  main_trainSet = pickle.load(open(ARGS.inputFileRadical+'.train', 'rb'))
  print("-> " + str(len(main_trainSet)) + " patients at dimension 0 for file: "+ ARGS.inputFileRadical + ".train dimensions ")
  main_testSet = pickle.load(open(ARGS.inputFileRadical+'.test', 'rb'))
  print("-> " + str(len(main_testSet)) + " patients at dimension 0 for file: "+ ARGS.inputFileRadical + ".test dimensions ")
  print("Note: these files carry 3D tensor data; the above numbers refer to dimension 0, dimensions 1 and 2 have irregular sizes.")

  ARGS.numberOfInputCodes = getNumberOfCodes([main_trainSet,main_testSet])
  print('Number of diagnosis input codes: ' + str(ARGS.numberOfInputCodes))

  #uses the same data for testing, but disregarding the fist admission of each patient
  labels_trainSet = pickle.load(open(ARGS.inputFileRadical+'.train', 'rb'))
  labels_testSet = pickle.load(open(ARGS.inputFileRadical+'.test', 'rb'))

  train_sorted_index = sorted(range(len(main_trainSet)), key=lambda x: len(main_trainSet[x]))  #lambda x: len(seq[x]) --> f(x) return len(seq[x])
  main_trainSet = [main_trainSet[i] for i in train_sorted_index]
  labels_trainSet = [labels_trainSet[i] for i in train_sorted_index]

  test_sorted_index = sorted(range(len(main_testSet)), key=lambda x: len(main_testSet[x]))
  main_testSet = [main_testSet[i] for i in test_sorted_index]
  labels_testSet = [labels_testSet[i] for i in test_sorted_index]

  trainSet = [main_trainSet, labels_trainSet]
  testSet = [main_testSet, labels_testSet]

  return trainSet, testSet

def performEvaluation(session, loss, x, y, mask, seqLen, test_Set, alignments_states):
  batchSize = ARGS.batchSize

  n_batches = int(np.ceil(float(len(test_Set[0])) / float(batchSize))) #default batch size is 100
  alignments_list2 = []
  alignments_list5 = []
  alignments_list10 = []
  crossEntropySum = 0.0
  dataCount = 0.0
  #computes de crossEntropy for all the elements in the test_Set, using the batch scheme of partitioning
  with session.as_default() as sess:
    for index in range(n_batches):
      batchX = test_Set[0][index * batchSize:(index + 1) * batchSize]
      batchY = test_Set[1][index * batchSize:(index + 1) * batchSize]
      xf, yf, maskf, nVisitsOfEachPatient_List = prepareHotVectors(batchX, batchY)

      # start_token = np.full((xf.shape[0], xf.shape[1], 1), 100)
      # xf = np.concatenate([start_token, xf], axis=-1)
      # yf = np.concatenate([start_token, yf], axis=-1)
      #
      # end_token = np.full((yf.shape[0], yf.shape[1], 1), 200)
      # yf = np.concatenate([yf, end_token], axis=-1)

      # np.concatenate([xf[-1], np.full((xf.shape[1], 1), -2)], axis=-1)

      feed_dict = {x: xf, y: yf, mask: maskf, seqLen: nVisitsOfEachPatient_List}
      crossEntropy = sess.run(loss, feed_dict=feed_dict)
      alignments = sess.run(alignments_states, feed_dict=feed_dict)

      if alignments[0].shape[-1] >= 2 and alignments[0].shape[-1] <= 10:
        alignments_list2.extend(alignments[0].tolist())

      if alignments[0].shape[-1] >= 10 and alignments[0].shape[-1] <= 20:
        alignments_list5.extend(alignments[0].tolist())

      if alignments[0].shape[-1] > 20:
        alignments_list10.extend(alignments[0].tolist())

      #accumulation by simple summation taking the batch size into account
      crossEntropySum += crossEntropy * len(batchX)
      dataCount += float(len(batchX))
      #At the end, it returns the mean cross entropy considering all the batches
  return n_batches, crossEntropySum / dataCount, [alignments_list2, alignments_list5, alignments_list10]

def testModel(session, predictions, x, y, mask, seqLen, test_Set):
  batchSize = ARGS.batchSize
  predictedY_list = []
  predictedProbabilities_list = []
  actualY_list = []
  predicted_yList = []
  n_batches = int(np.ceil(float(len(test_Set[0])) / float(batchSize))) #default batch size is 100

  with session.as_default() as sess:
    for index in range(n_batches):
      batchX = test_Set[0][index * batchSize:(index + 1) * batchSize]
      batchY = test_Set[1][index * batchSize:(index + 1) * batchSize]
      xf, yf, maskf, nVisitsOfEachPatient_List = prepareHotVectors(batchX, batchY)
      maxNumberOfAdmissions = np.max(nVisitsOfEachPatient_List)

      predicted_y = sess.run(predictions, feed_dict={x: xf, y: yf, mask: maskf, seqLen: nVisitsOfEachPatient_List})
      predicted_yList.append(predicted_y.tolist()[-1])

      # traverse the predicted results, once for each patient in the batch
      for ith_patient in range(predicted_y.shape[1]):
        predictedPatientSlice = predicted_y[:, ith_patient, :]
        # retrieve actual y from batch tensor -> actual codes, not the hotvector
        actual_y = batchY[ith_patient][1:]
        # for each admission of the ith-patient
        for ith_admission in range(nVisitsOfEachPatient_List[ith_patient]):
          # convert array of actual answers to list
          actualY_list.append(actual_y[ith_admission])
          # retrieves ith-admission of ths ith-patient
          ithPrediction = predictedPatientSlice[ith_admission]
          # since ithPrediction is a vector of probabilties with the same dimensionality of the hotvectors
          # enumerate is enough to retrieve the original codes
          enumeratedPrediction = [codeProbability_pair for codeProbability_pair in enumerate(ithPrediction)]
          # sort everything
          sortedPredictionsAll = sorted(enumeratedPrediction, key=lambda x: x[1], reverse=True)
          # creates trimmed list up to max(maxNumberOfAdmissions,30) elements
          sortedTopPredictions = sortedPredictionsAll[0:max(maxNumberOfAdmissions, 30)]
          # here we simply toss off the probability and keep only the sorted codes
          sortedTopPredictions_indexes = [codeProbability_pair[0] for codeProbability_pair in sortedTopPredictions]
          # stores results in a list of lists - after processing all batches, predictedY_list stores all the prediction results
          predictedY_list.append(sortedTopPredictions_indexes)
          predictedProbabilities_list.append(sortedPredictionsAll)

    # ---------------------------------Report results using k=[10,20,30]
    print('==> computation of prediction results with constant k')
    recall_sum = [0.0, 0.0, 0.0]

    k_list = [10, 20, 30]
    for ith_admission in range(len(predictedY_list)):
      ithActualYSet = set(actualY_list[ith_admission])
      for ithK, k in enumerate(k_list):
        ithPredictedY = set(predictedY_list[ith_admission][:k])
        intersection_set = ithActualYSet.intersection(ithPredictedY)
        recall_sum[ithK] += len(intersection_set) / float(
          len(ithActualYSet))  # this is recall because the numerator is len(ithActualYSet)

    precision_sum = [0.0, 0.0, 0.0]
    k_listForPrecision = [1, 2, 3]
    for ith_admission in range(len(predictedY_list)):
      ithActualYSet = set(actualY_list[ith_admission])
      for ithK, k in enumerate(k_listForPrecision):
        ithPredictedY = set(predictedY_list[ith_admission][:k])
        intersection_set = ithActualYSet.intersection(ithPredictedY)
        precision_sum[ithK] += len(intersection_set) / float(
          k)  # this is precision because the numerator is k \in [10,20,30]

    finalRecalls = []
    finalPrecisions = []
    for ithK, k in enumerate(k_list):
      finalRecalls.append(recall_sum[ithK] / float(len(predictedY_list)))
      finalPrecisions.append(precision_sum[ithK] / float(len(predictedY_list)))

    print('Results for Recall@' + str(k_list))
    print(str(finalRecalls[0]))
    print(str(finalRecalls[1]))
    print(str(finalRecalls[2]))

    print('Results for Precision@' + str(k_listForPrecision))
    print(str(finalPrecisions[0]))
    print(str(finalPrecisions[1]))
    print(str(finalPrecisions[2]))

  return


def decoderCell(inputs, lengths):
  # inputs = tf.transpose(inputs, [1,0,2])
  attention_mechanism = tf.contrib.seq2seq.BahdanauMonotonicAttention(ARGS.attentionDimSize, memory=inputs, memory_sequence_length=lengths)

  lstms = [tf.nn.rnn_cell.LSTMCell(size) for size in ARGS.hiddenDimSize]  # According to docs (https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/rnn_cell/LSTMCell), the peephole version is based on LSTM Google (2014)
  lstms = [tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=(1-ARGS.dropoutRate)) for lstm in lstms]
  dec_cell = tf.nn.rnn_cell.MultiRNNCell(lstms)
  dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attention_mechanism, alignment_history=True)
  return dec_cell

def EncoderDecoderAttention_layer(inputTensor, targetTensor, seqLen):
  inputTensor = tf.transpose(inputTensor, [1,0,2])
  # Encoder
  with tf.variable_scope('encoder'):
    lstms = [tf.nn.rnn_cell.LSTMCell(size) for size in ARGS.hiddenDimSize] #According to docs (https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/rnn_cell/LSTMCell), the peephole version is based on LSTM Google (2014)
    lstms = [tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=(1-ARGS.dropoutRate)) for lstm in lstms]
    enc_cell = tf.nn.rnn_cell.MultiRNNCell(lstms)
    encoder_outputs, encoder_states = tf.nn.dynamic_rnn(enc_cell, inputTensor, sequence_length=seqLen, time_major=False, dtype=tf.float32)

  dec_start_state = tuple(encoder_states[-1] for _ in range(len(ARGS.hiddenDimSize)))
  seqLen = tf.cast(seqLen, dtype=tf.int32)

  go_token = 2.
  end_token = 3.

  # go_tokens = tf.fill((1, tf.shape(targetTensor)[0], ARGS.numberOfInputCodes), go_token)
  # end_tokens = tf.fill((tf.shape(targetTensor)[0], tf.shape(targetTensor)[1], ARGS.numberOfInputCodes), end_token)
  # dec_input = tf.concat([go_tokens, targetTensor], axis=1)
  # dec_input = tf.concat([targetTensor, end_tokens], axis=1)
  # dec_input = tf.concat([tf.strided_slice(dec_input, begin=[-1, 0, 0], end=[-1, tf.shape(dec_input)[1], ARGS.numberOfInputCodes], strides=[1,1,1]), end_tokens], axis=-1)

  with tf.variable_scope('decoder'):
    # Decoder
    dec_cell = decoderCell(encoder_outputs, seqLen)
    init_state = dec_cell.zero_state(tf.shape(targetTensor)[1], tf.float32)
    init_state = init_state.clone(cell_state=dec_start_state)

    helper = tf.contrib.seq2seq.TrainingHelper(inputs=targetTensor, sequence_length=seqLen, time_major=False)
    decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell, helper=helper, initial_state=init_state)

    training_outputs, training_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False, training=True)

  tiled_start_state = tf.contrib.seq2seq.tile_batch(dec_start_state, multiplier=ARGS.beamWidth)
  tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=ARGS.beamWidth)
  tiled_lengths = tf.contrib.seq2seq.tile_batch(seqLen, multiplier=ARGS.beamWidth)

  with tf.variable_scope('decoder', reuse=True):
    dec_cell = decoderCell(tiled_encoder_outputs, tiled_lengths)
    # init_state = dec_cell.zero_state(tf.shape(targetTensor)[0] * tf.shape(targetTensor)[1] * beam_width * beam_width//tf.shape(targetTensor)[0], tf.float32)
    init_state = dec_cell.zero_state(tf.shape(targetTensor)[1] * ARGS.beamWidth, tf.float32)
    init_state = init_state.clone(cell_state=tiled_start_state)

    go_token = tf.cast(go_token, dtype=tf.int32)
    end_token = tf.cast(end_token, dtype=tf.int32)

    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
      cell=dec_cell,
      embedding=tf.Variable(tf.zeros([ARGS.hiddenDimSize[-1], ARGS.numberOfInputCodes])),
      start_tokens=tf.ones_like(seqLen) * go_token,
      end_token=end_token,
      initial_state=init_state,
      beam_width=ARGS.beamWidth)

    inference_outputs, inference_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder, output_time_major=False, maximum_iterations=ARGS.maxDecoderIterations, training=False)

  if ARGS.state == "cell":
    # inference_outputs = tf.transpose(inference_state.cell_state.cell_state[-1].c, [1,0,2])
    inference_outputs = tf.transpose(inference_state[0].attention, [1,0,2])
    # inference_outputs = tf.transpose(inference_state[0].cell_state[-1].c, [1,0,2])
  elif ARGS.state == "hidden":
    inference_outputs = tf.transpose(inference_state.cell_state.cell_state[-1].h, [1,0,2])
  elif ARGS.state == "attention":
    inference_outputs = tf.transpose(inference_state.cell_state.attention, [1,0,2])
  else:
    inference_outputs = tf.cast(inference_outputs.predicted_ids, tf.float32)

  return inference_outputs, inference_state[0].alignment_history


def FC_layer(inputTensor):
  im_dim = inputTensor.get_shape()[-1]
  weights = tf.get_variable(name='weights',
                               shape=[im_dim, ARGS.numberOfInputCodes],
                               dtype=tf.float32,
                               initializer=tf.keras.initializers.glorot_normal())

  bias = tf.get_variable(name='bias',
                           shape=[ARGS.numberOfInputCodes],
                           dtype=tf.float32,
                           initializer=tf.zeros_initializer())
  output = tf.nn.leaky_relu(tf.add(tf.matmul(inputTensor, weights), bias))
  return output, weights, bias

def build_model():
  graph = tf.Graph()
  with graph.as_default():
    xf = tf.placeholder(tf.float32, [None, None, ARGS.numberOfInputCodes], name="inputs")
    yf = tf.placeholder(tf.float32, [None, None, ARGS.numberOfInputCodes], name="labels")
    maskf = tf.placeholder(tf.float32, [None, None], name="mask")
    seqLen = tf.placeholder(tf.float32, [None], name="nVisitsOfEachPatient_List")

    with tf.device('/gpu:0'):
      flowingTensor, alignments_states = EncoderDecoderAttention_layer(xf, yf, seqLen)
      flowingTensor, weights, bias = FC_layer(flowingTensor)
      flowingTensor = tf.nn.softmax(flowingTensor)
      flowingTensor = tf.math.multiply(flowingTensor, maskf[:,:,None], name="predictions")

      epislon = 1e-8
      cross_entropy = -(yf * tf.math.log(flowingTensor + epislon) + (1. - yf) * tf.math.log(1. - flowingTensor + epislon))
      prediction_loss = tf.math.reduce_mean(tf.math.reduce_sum(cross_entropy, axis=[2, 0]) / seqLen)
      L2_regularized_loss = prediction_loss + ARGS.LregularizationAlpha * tf.nn.l2_loss(weights)

      # optimizer = tf.train.AdadeltaOptimizer(learning_rate=ARGS.learningRate, rho=0.95, epsilon=1e-06).minimize(L2_regularized_loss)
      # optimizer = tf.train.AdamOptimizer(learning_rate=ARGS.learningRate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(L2_regularized_loss)
      optimizer = tf.train.RMSPropOptimizer(learning_rate=ARGS.learningRate, decay=ARGS.decay, momentum=ARGS.momentum, epsilon=1e-10).minimize(L2_regularized_loss)

    return tf.global_variables_initializer(), graph, optimizer, L2_regularized_loss, xf, yf, maskf, seqLen, flowingTensor, alignments_states

def train_model():
  print("==> data loading")
  trainSet, testSet = load_data()

  print("==> model building")
  init, graph, optimizer, loss, x, y, mask, seqLen, predictions, alignments_states = build_model()

  print ("==> training and validation")
  batchSize = ARGS.batchSize
  n_batches = int(np.ceil(float(len(trainSet[0])) / float(batchSize)))

  bestAlignments = []
  bestValidationCrossEntropy = 1e20
  bestValidationEpoch = 0
  bestModelDirName = ''

  iImprovementEpochs = 0
  iConsecutiveNonImprovements = 0
  epoch_counter = 0

  with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(init)

    for epoch_counter in range(ARGS.nEpochs):
      iteration = 0
      trainCrossEntropyVector = []
      for index in random.sample(range(n_batches), n_batches):
        batchX = trainSet[0][index*batchSize:(index+1)*batchSize]
        batchY = trainSet[1][index*batchSize:(index + 1)*batchSize]
        xf, yf, maskf, nVisitsOfEachPatient_List = prepareHotVectors(batchX, batchY)
        xf += np.random.normal(0, 0.1, xf.shape)

        feed_dict = {x: xf, y: yf, mask: maskf, seqLen: nVisitsOfEachPatient_List}
        _, trainCrossEntropy = sess.run([optimizer, loss], feed_dict=feed_dict)

        trainCrossEntropyVector.append(trainCrossEntropy)
        iteration += 1

      print('-> Epoch: %d, mean cross entropy considering %d TRAINING batches: %f' % (epoch_counter, n_batches, np.mean(trainCrossEntropyVector)))
      nValidBatches, validationCrossEntropy, testAlignments = performEvaluation(sess, loss, x, y, mask, seqLen, testSet, alignments_states)
      print('      mean cross entropy considering %d VALIDATION batches: %f' % (nValidBatches, validationCrossEntropy))

      if validationCrossEntropy < bestValidationCrossEntropy:
        iImprovementEpochs += 1
        iConsecutiveNonImprovements = 0
        bestValidationCrossEntropy = validationCrossEntropy
        bestValidationEpoch = epoch_counter

        if os.path.exists(bestModelDirName):
          shutil.rmtree(bestModelDirName)
        bestModelDirName = ARGS.outFile + '.' + str(epoch_counter)

        if os.path.exists(bestModelDirName):
          shutil.rmtree(bestModelDirName)

        signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs= {"inputs": x, "labels": y, "mask": mask, "seqLen": seqLen}, outputs= {"predictions": predictions})
        builder = tf.saved_model.builder.SavedModelBuilder(bestModelDirName)
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={'model': signature})
        builder.save()
        bestAlignments = testAlignments

      else:
        print('Epoch ended without improvement.')
        iConsecutiveNonImprovements += 1
      if iConsecutiveNonImprovements > ARGS.maxConsecutiveNonImprovements: #default is 10
        break

    # Best results
    print('--------------SUMMARY--------------')
    print('The best VALIDATION cross entropy occurred at epoch %d, the value was of %f ' % (
    bestValidationEpoch, bestValidationCrossEntropy))
    print('Best model file: ' + bestModelDirName)
    print('Number of improvement epochs: ' + str(iImprovementEpochs) + ' out of ' + str(epoch_counter + 1) + ' possible improvements.')
    print('Note: the smaller the cross entropy, the better.')
    print('-----------------------------------')
    
    sess.close()
    return

    for array in bestAlignments[0]:
      array.insert(0,0)

    bestAlignments2 = np.array(list(zip_longest(*bestAlignments[0], fillvalue=0))).T
    bestAlignments2 = np.mean(bestAlignments2, axis=0)

    for array in bestAlignments[1]:
      array.insert(0,0)

    bestAlignments5 = np.array(list(zip_longest(*bestAlignments[1], fillvalue=0))).T
    bestAlignments5 = np.mean(bestAlignments5, axis=0)

    for array in bestAlignments[2]:
      array.insert(0,0)

    bestAlignments10 = np.array(list(zip_longest(*bestAlignments[2], fillvalue=0))).T
    bestAlignments10 = np.mean(bestAlignments10, axis=0)

    # testModel(sess, predictions, x, y, mask, seqLen, testSet)

    if bestAlignments2 is not None:
      plt.plot(bestAlignments2, "darkorange", label='2 to 10 admissions')
    if bestAlignments5 is not None:
      plt.plot(bestAlignments5, "indigo", label='2 to 11 to 2 to 20 admissions')
    if bestAlignments10 is not None:
      plt.plot(bestAlignments10, "seagreen", label='2 to more than 20 admissions')
    
    # plt.xticks(range(0,len(bestAlignments10)))
    plt.gca().xaxis.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    # plt.tick_params(axis='x', which='major', labelsize=5)
    plt.locator_params(axis="x", nbins=30)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.ylabel("Alignment Score")
    plt.xlabel("Amount of Admissions")
    plt.savefig('alignments.png', dpi=300, bbox_inches="tight")

    # plt.show()

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('inputFileRadical', type=str, metavar='<visit_file>', help='File radical name (the software will look for .train and .test files) with pickled data organized as patient x admission x codes.')
  parser.add_argument('outFile', metavar='out_file', default='model_output', help='Any file directory to store the model.')
  parser.add_argument('--maxConsecutiveNonImprovements', type=int, default=10, help='Training wiil run until reaching the maximum number of epochs without improvement before stopping the training')
  parser.add_argument('--hiddenDimSize', type=str, default='[542]', help='Number of layers and their size - for example [100,200] refers to two layers with 100 and 200 nodes.')
  parser.add_argument('--state', type=str, default='cell', help='Pass cell, hidden or attention to fully connected layer')
  parser.add_argument('--attentionDimSize', type=int, default=5, help='Number of attention layer dense units')
  parser.add_argument('--maxDecoderIterations', type=int, default=1, help='Maximum Inference Decoder iterations over predicted data')
  parser.add_argument('--beamWidth', type=int, default=1, help='Beam width size')
  parser.add_argument('--batchSize', type=int, default=100, help='Batch size.')
  parser.add_argument('--nEpochs', type=int, default=1000, help='Number of training iterations.')
  parser.add_argument('--LregularizationAlpha', type=float, default=0.001, help='Alpha regularization for L2 normalization')
  parser.add_argument('--learningRate', type=float, default=0.001, help='Learning rate.')
  parser.add_argument('--decay', type=float, default=0.9, help='Decay.')
  parser.add_argument('--momentum', type=float, default=0.1, help='Momentum.')
  # parser.add_argument('--globalStep', type=float, default=0, help='Global step.')
  # parser.add_argument('--decaySteps', type=float, default=1000, help='Decay steps.')
  # parser.add_argument('--decayRate', type=float, default=0.7, help='Decay rate.')
  parser.add_argument('--dropoutRate', type=float, default=0.1, help='Dropout probability.')

  ARGStemp = parser.parse_args()
  hiddenDimSize = [int(strDim) for strDim in ARGStemp.hiddenDimSize[1:-1].split(',')]
  ARGStemp.hiddenDimSize = hiddenDimSize
  return ARGStemp

if __name__ == '__main__':
  global ARGS
  ARGS = parse_arguments()

  train_model()