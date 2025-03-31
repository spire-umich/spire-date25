import argparse, pickle, spire
import numpy as np
from collections import defaultdict

def main():
  # Call train() or analyze() based on the user's command
  args = processArguments()
  args.func(args)


def train(args):
  # Import all training samples
  importedSamples, timeName, workName = importMultipleSampleCsvs(args.csvs)
  mergedSamples, sampleCount = mergeSampleDicts(importedSamples.values())
  del importedSamples

  print("\nTraining data:")
  print(f"  Time is \"{timeName}\"")
  print(f"  Work is \"{workName}\"")
  print(f"  {len(mergedSamples):,} Metrics")
  print(f"  {sampleCount:,} Samples")

  # Train a SPIRE model and save it
  print("\nTraining SPIRE model...")
  model = spire.Ensemble().fit(mergedSamples)
  saveModel(args.model, model, timeName, workName)


def analyze(args):
  # Load the SPRIE model
  model, timeName, workName = loadModel(args.model)
  print("")

  # Import the CSVs to be analyzed
  importedSamples, importedTimeName, importedWorkName = importMultipleSampleCsvs(args.csvs)
  if timeName != importedTimeName:
    raise ValueError(f"The model's Time name doesn't match the CSV Time name (\"{timeName}\" and "
                     f"\"{importedTimeName}\", respectively)")
  if workName != importedWorkName:
    raise ValueError(f"The model's Work name doesn't match the CSV Work name (\"{workName}\" and "
                     f"\"{importedWorkName}\", respectively)")

  # Analyze each CSV's samples
  for path in args.csvs:
    results = model.predict(importedSamples[path])
    print(f"\n{args.n} Lowest-Throughput Metric(s) for \"{path}\":")
    for metricName, P_pred, timeLeft in results[:args.n]:
      print(f"  {P_pred:0.2f}  {timeLeft:0.2f}  {metricName}")
    print("")


def processArguments():
  parser = argparse.ArgumentParser()
  fc = argparse.ArgumentDefaultsHelpFormatter
  s  = parser.add_subparsers(dest="command", required=True)

  # "train" command and arguments
  p = s.add_parser("train", help="train a SPIRE model", formatter_class=fc)
  p.add_argument("model", help="output file for the trained model")
  p.add_argument("csvs",  help="CSV file containing training samples",   metavar="csv", nargs="+")
  # Setting the default for "func" so train() is called
  p.set_defaults(func=train)
  
  # "analyze" command and arguments
  p = s.add_parser("analyze", help="analyze with a trained SPIRE model", formatter_class=fc)
  p.add_argument("model", help="file containing a trained model")
  p.add_argument("csvs",  help="CSV file containing samples to analyze", metavar="csv", nargs="+")
  p.add_argument("-n",    help="report the N lowest-throughput metrics", metavar="N",
                          type=int, default=10)
  # Setting the default for "func" so analyze() is called
  p.set_defaults(func=analyze)
  
  return parser.parse_args()


def importSampleCsv(path):
  print(f"Importing \"{path}\"...")
  
  # Read the CSV file
  with open(path, "r") as fin:
    args = dict(delimiter=",",   comments=None, deletechars=None, replace_space=None,
                defaultfmt=None, loose=False)
    header = np.genfromtxt(fin, **args, dtype=str, max_rows=1)
    data   = np.genfromtxt(fin, **args, dtype=np.double)
    del args
  if len(data) == 0:
    raise ValueError(f"CSV file contains no samples")

  # data will be 1D if only 1 row of samples was read. Make it 2D for consistency.
  assert data.ndim <= 2, "Assuming genfromtxt won't return an array >2D"
  if data.ndim == 1:
    data = data.reshape(1, -1)

  # Check the column count. Each metric should have 3 columns (time, work, metric).
  assert header.shape[0] == data.shape[1], "Assuming the two reads found the same number of columns"
  if header.shape[0] % 3 != 0:
    raise ValueError(f"CSV files must contain a multiple of 3 columns (columns = {header.shape[0]})")
  
  # Get names from the header row and check for empty names
  timeName    = str(header[0])
  workName    = str(header[1])
  metricNames = header[2::3]

  if len(timeName) == 0:
    raise ValueError("Missing Time's name in row 1, column 1")
  if len(workName) == 0:
    raise ValueError("Missing Work's name in row 1, column 2")
  
  lens = np.char.str_len(metricNames)
  idx  = np.argmin(lens)
  if lens[idx] == 0:
    raise ValueError(f"Missing Metric's name in row 1, column {3 * idx + 3}")
  del lens, idx

  # Check for duplicate metrics
  unique, count = np.unique(metricNames, return_counts=True)
  idx = np.argmax(count)
  if count[idx] > 1:
    raise ValueError(f"A Metric's name is repeated: \"{unique[idx]}\"")
  del unique, count, idx

  # Collect all samples in a dictionary keyed by metric name
  samples = {}
  sampleCount = 0
  for i, metricName in enumerate(metricNames):
    # Each metric's samples span 3 columns (work, time, metric value)
    columns = data[:, (3 * i):(3 * i + 3)]

    # Drop samples that contain NaN for any of its 3 values
    # This includes empty cells that were assigned NaN by genfromtxt
    noNans  = ~np.isnan(columns).any(axis=1)
    samples[str(metricName)] = columns[noNans]
    sampleCount += np.count_nonzero(noNans)

  print(f"  Imported {sampleCount:,} samples for {len(samples):,} metrics")
  return samples, timeName, workName


def importMultipleSampleCsvs(paths):
  results  = {}
  timeName = None
  workName = None
  if len(paths) == 0:
    return results, timeName, workName

  # The first CSV's Work and Time names will be the baseline for comparison
  samples, timeName, workName = importSampleCsv(paths[0])
  results[paths[0]] = samples

  # Import the remaining CSVs
  for path in paths[1:]:
    # Skip importing redundant CSVs
    if path in results:
      continue

    samples, fileTimeName, fileWorkName = importSampleCsv(path)
    if fileTimeName != timeName:
      raise ValueError(f"Time's name is inconsistent (\"{paths[0]}\" had \"{timeName}\" but "
                       f"\"{path}\" had \"{fileTimeName}\")")
    if fileWorkName != workName:
      raise ValueError(f"Work's name is inconsistent (\"{paths[0]}\" had \"{workName}\" but "
                       f"\"{path}\" had \"{fileWorkName}\")")
    results[path] = samples

  return results, timeName, workName


def mergeSampleDicts(unmergedSamples):
  # Collect a list of sample arrays for each metric
  dictOfLists = defaultdict(list)
  for sampleDict in unmergedSamples:
    for metricName, metricSamples in sampleDict.items():
      dictOfLists[metricName].append(metricSamples)

  # Stack all sample arrays for each metric in one vstack call
  result = {}
  sampleCount = 0
  for metricName, listOfArrays in dictOfLists.items():
    temp = np.vstack(listOfArrays)
    result[metricName] = temp
    sampleCount += temp.shape[0]
  return result, sampleCount


# Note: this uses pickle which has security and portability limitations
# See:  https://docs.python.org/3/library/pickle.html
def saveModel(path, model, timeName, workName):
  print(f"Saving model to \"{path}\"...")
  with open(path, "wb") as fout:
    pickle.dump(model,    fout)
    pickle.dump(timeName, fout)
    pickle.dump(workName, fout)


# Note: this uses pickle which has security and portability limitations
# See:  https://docs.python.org/3/library/pickle.html
def loadModel(path):
  print(f"Loading model from \"{path}\"...")
  with open(path, "rb") as fin:
    model    = pickle.load(fin)
    timeName = pickle.load(fin)
    workName = pickle.load(fin)
  print(f"  Time is \"{timeName}\"")
  print(f"  Work is \"{workName}\"")
  print(f"  {len(model):,} Metrics")
  return model, timeName, workName


if __name__ == "__main__":
  main()