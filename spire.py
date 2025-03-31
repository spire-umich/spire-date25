import numpy as np
import networkx as nx

# ----------------------------------------------------------------------------------------------- #
# Ensemble - Implements a full SPIRE model (an *ensemble* of roofline models)
# ----------------------------------------------------------------------------------------------- #
class Ensemble:
  def __init__(self):
    self._fitted = False


  def fit(self, samples):
    # Create a roofline for each metric
    self._rooflines = {}
    for metricName, metricSamples in samples.items():
      try:
        T, W, M = Ensemble._preprocessInputs(metricSamples)

        # Compute I (Operational Intensity) and P (Throughput) for each sample
        # Suppressing division warnings and converting NaNs (0/0) to 0
        with np.errstate(divide="ignore", invalid="ignore"):
          I = np.nan_to_num(W / M, nan=0)
          P = np.nan_to_num(W / T, nan=0)
        del T, W, M

        # Fit the metric's roofline and save it
        roofline = Roofline().fit(I, P)
        self._rooflines[metricName] = roofline
      except Exception as e:
        # To aid with debugging, report which metric caused the exception
        raise Exception(f"An exception was raised while fitting metric \"{metricName}\"") from e

    self._fitted = True
    return self


  def predict(self, samples):
    if not self._fitted:
      raise Exception(f"This {self.__class__.__name__} instance is not fitted yet")
    
    # Check that all, and only, the expected metrics are present
    extraMetrics = samples.keys() - self._rooflines.keys()
    if len(extraMetrics):
      example = next(iter(extraMetrics))
      raise ValueError(f"\"samples\" contains {len(extraMetrics)} metric(s) not present "
                       f"during fitting (e.g., \"{example}\")")
    
    missingMetrics = self._rooflines.keys() - samples.keys()
    if len(missingMetrics):
      example = next(iter(missingMetrics))
      raise ValueError(f"\"samples\" is missing {len(missingMetrics)} metric(s) present "
                       f"during fitting (e.g., \"{example}\")")
    del extraMetrics, missingMetrics

    # Collect prediction results for each metric
    results = []
    for metricName, metricSamples in samples.items():
      try:
        T, W, M = Ensemble._preprocessInputs(metricSamples)

        # Compute I (Operational Intensity) for each sample
        # Suppressing division warnings and converting NaNs (0/0) to 0
        with np.errstate(divide="ignore", invalid="ignore"):
          I = np.nan_to_num(W / M, nan=0)
        del W, M

        # Predict P (Throughput) for each sample
        # Merge these predictions with a time-weighted average
        P_pred = self._rooflines[metricName].predict(I)
        P_pred = np.sum(T * P_pred) / np.sum(T)

        # Compute the fraction of time samples were left of the roofline's peak
        isLeft   = (I <= self._rooflines[metricName].getPeakI())
        timeLeft = np.sum(T[isLeft]) / np.sum(T)

        # Collect this metric's results
        results.append((metricName, P_pred, timeLeft))
      except Exception as e:
        # To aid with debugging, report which metric caused the exception
        raise Exception(f"An exception was raised while predicting with metric \"{metricName}\"") from e
    
    # Sort the results by predicted throughput in ascending order
    results.sort(key=lambda x: x[1])

    return results
  

  def __len__(self):
    return len(self._rooflines)

  def __contains__(self, metricName):
    return metricName in self._rooflines

  def __getitem__(self, metricName):
    # Returns a reference to the roofline (not a copy)
    return self._rooflines[metricName]

  def __iter__(self):
    # Returns references to the rooflines (not copies)
    return self._rooflines.items()
  
  def rooflines(self):
    return iter(self)
  
  def metrics(self):
    return self._rooflines.keys()


  @staticmethod
  def _preprocessInputs(metricSamples):
    temp = _convertToF64Array(metricSamples, f"samples[metricName]")
    if temp.ndim != 2:
      raise ValueError(f"samples[metricName] must be 2D (samples[metricName].ndim = {temp.ndim})")
    if temp.shape[1] != 3:
      raise ValueError(f"samples[metricName] must have 3 columns "
                       f"(samples[metricName].shape[1] = {temp.shape[1]})")
    if temp.shape[0] == 0:
      raise ValueError(f"samples[metricName] cannot be empty")

    # Get Time, Work, Metric arrays from the first 3 columns
    T = temp[:, 0]  # Time
    W = temp[:, 1]  # Work
    M = temp[:, 2]  # Metric

    # T, W, M must be >= 0 and finite
    _validateArrayDefault(T, False, "T (Time)")
    _validateArrayDefault(W, False, "W (Work)")
    _validateArrayDefault(M, False, "M (Metric)")
    # If T is 0, no time was spent collecting the sample
    # Thus, if T is 0, W and M should be 0
    if ((T == 0) & ((W != 0) | (M != 0))).any():
      raise ValueError("If T (Time) is 0, W (Work) and M (Metric) should be 0")

    return T, W, M

# ----------------------------------------------------------------------------------------------- #
# Roofline - Implements a single SPIRE roofline model
# ----------------------------------------------------------------------------------------------- #
class Roofline:
  def __init__(self):
    self._fitted = False


  def fit(self, I, P):
    I, P = self._preprocessFitInputs(I, P)

    I_fitL, P_fitL = Roofline._fitLeft(I, P)
    I_fitR, P_fitR = Roofline._fitRight(I, P)

    # Sanity checks for fitting results
    maxP = np.max(P)
    assert P_fitL[-1] == maxP,      "_fitLeft's last point should have p = max(P))"
    assert P_fitR[ 0] == maxP,      "_fitRight's first point should have p = max(P)"
    assert I_fitL[-1] <= I_fitR[0], "_fitLeft's last point should be left of (or same as) " \
                                    "_fitRight's first point"

    # If both fits reached the same highest point,
    if I_fitL[-1] == I_fitR[0]:
      # Skip the redundant point
      self._I_fit = np.hstack((I_fitL[:-1], I_fitR))
      self._P_fit = np.hstack((P_fitL[:-1], P_fitR))

      # Use the common point as the roofline's peak location
      self._peakI = I_fitL[-1]
    else:
      self._I_fit = np.hstack((I_fitL, I_fitR))
      self._P_fit = np.hstack((P_fitL, P_fitR))

      # Use the midpoint as the roofline's peak location
      self._peakI = (I_fitL[-1] + I_fitR[0]) / 2

    # Precompute the line segment paramters used during prediction
    self._precomputeSegmentParams()

    self._fitted = True
    return self
  

  def predict(self, I):
    if not self._fitted:
      raise Exception(f"This {self.__class__.__name__} instance is not fitted yet")
    I = _convertToF64Array(I, "I")

    # I should be >= 0, but it can be infinite
    _validateArrayDefault(I, True, "I (Operational Intensity)")

    # Identify which segment each I falls under
    # If I[x] <  the first segment, seg[x] = -1
    # If I[x] >= the last  segment, seg[x] = len(I_fit) - 1
    seg = np.digitize(I, self._I_fit) - 1

    # Compute the predicted P using the appropriate parameters
    P = self._M[seg] * I + self._B[seg]
    
    # Use the last point's P if I is right of the last segment
    P[seg == len(self._I_fit) - 1] = self._P_fit[-1]

    # If I > 0, I[x] should never be left of the first segment
    assert (seg != -1).all(), "Assuming I[x] will never be < the first segment"
    # If we wanted to handle this special case like the other one:
    # P[seg == -1] = self._P_fit[0]

    return P


  def getFitSegments(self):
    if not self._fitted:
      raise Exception(f"This {self.__class__.__name__} instance is not fitted yet")
    # Returning copies to protect _I_fit and _P_fit from modification
    return np.copy(self._I_fit), np.copy(self._P_fit)


  def getPeakI(self):
    if not self._fitted:
      raise Exception(f"This {self.__class__.__name__} instance is not fitted yet")
    # Returning a copy to protect _peakI from modification
    return self._peakI.copy()


  def _precomputeSegmentParams(self):
    # Segment slopes: m = (y1 - y0) / (x1 - x0)
    self._M = np.empty_like(self._I_fit)
    dx = np.diff(self._I_fit)
    assert (dx != 0).all(), "All fit points should have unique x values"
    self._M[:-1] = np.diff(self._P_fit) / dx

    # Segment intercepts: b = y0 - m * x0
    self._B = np.empty_like(self._I_fit)
    self._B[:-1] = self._P_fit[:-1] - self._M[:-1] * self._I_fit[:-1]

    # These will be used when I is outside the piecewise segments
    self._M[-1] = np.nan
    self._B[-1] = np.nan


  @staticmethod
  def _preprocessFitInputs(I, P):
    I = _convertToF64Array(I, "I")
    P = _convertToF64Array(P, "P")
    
    # Validate I and P array properties
    if I.ndim != 1 or P.ndim != 1:
      raise ValueError(f"I and P must be 1D (I.ndim = {I.ndim}, P.ndim = {P.ndim})")
    if len(I) != len(P):
      raise ValueError(f"I and P must be the same length (len(I) = {len(I)}, len(P) = {len(P)})")
    if len(I) == 0:
      raise ValueError(f"I and P cannot be empty")

    # Validate I and P contents
    # I must be >= 0
    # P must be >= 0 and finite
    _validateArrayDefault(I, True,  "I (Operational Intensity)")
    _validateArrayDefault(P, False, "P (Throughput)")
    # Assuming M and T are finite, I[x] and P[x] are 0 iff W[x] is 0
    # Thus, I[x] is 0 iff P[x] is 0
    if ((I == 0) != (P == 0)).any():
      raise ValueError(f"I[x] (Operational Intensity) should be 0 iff P[x] (Throughput) is 0")

    return I, P


  @staticmethod
  def _fitLeft(I, P):
    # Start at the origin
    fitPoints = [(0, 0)]

    while True:
      # Compute the slope from the previous fit point to all points to its right
      # (points to its left will be given slopes of -infinity)
      prev    = fitPoints[-1]
      isRight = (prev[0] < I)
      slope   = np.full(len(I), -np.inf)
      slope[isRight] = (P[isRight] - prev[1]) / (I[isRight] - prev[0])

      # Identify the highest-slope point. If its slope is <= 0, we're done fitting.
      idx = np.argmax(slope)
      if slope[idx] <= 0:
        break

      # Add the highest-slope point to the fit point list (and repeat)
      fitPoints.append((I[idx], P[idx]))

    # Convert the fit point list into 2 arrays
    fitPoints = np.array(fitPoints)
    I_fit = fitPoints[:, 0]
    P_fit = fitPoints[:, 1]
    return I_fit, P_fit


  @staticmethod
  def _fitRight(I, P):
    # Identify the Pareto points and ignore the rest
    # These will be returned sorted right to left
    I, P = Roofline._getSortedParetoPoints(I, P)

    # If there isn't a starting point "s" at I = infinity, add a temporary one
    addedS = False
    if I[0] != np.inf:
      addedS = True
      I = np.insert(I, 0, np.inf)     # s is the rightmost point
      P = np.insert(P, 0, P[0] - 1)   # s is the lowest point

    # Create the DAG and find the shortest path from "Start" to "End"
    graph, hDict = Roofline._getFittingGraph(I, P)
    path = nx.dijkstra_path(graph, "Start", "End", weight="weight")

    # Convert the path's vertices into fitting points
    I_fit, P_fit = Roofline._getPathPoints(I, P, hDict, path)

    # Reverse the points to order them left to right
    # If we added a temporary starting point, drop it
    if addedS:
      I_fit = I_fit[:0:-1]
      P_fit = P_fit[:0:-1]
    else:
      I_fit = I_fit[::-1]
      P_fit = P_fit[::-1]

    return I_fit, P_fit


  @staticmethod
  def _getSortedParetoPoints(I, P):
    # Sort points by I (descending/right to left)
    # For ties, sort by P (descending/top to bottom)
    indices  = np.lexsort((-P, -I))
    P_sorted = P[indices]

    # Visit points in sorted order (right to left, top to bottom)
    # If a point has the highest P seen thus far, it's a Pareto point
    cMax = np.hstack((-np.inf, np.maximum.accumulate(P_sorted[:-1])))
    isPareto = (P_sorted > cMax)

    # Collect the sorted Pareto points based on the isPareto mask
    I_pareto = I[indices[isPareto]]
    P_pareto = P_sorted[isPareto]

    return I_pareto, P_pareto


  # Optimization:
  # Many parts of this graph won't be visited during Dijkstra's. Instead of constructing the whole
  # thing up front, compute edges and verticies on the fly as Dijkstra's progresses. All we need
  # to compute (x, y)'s neighbors and edge weights are the Pareto points and the slope of xy.
  # Both of these items will be available when Dijkstra's visits (x, y).
  @staticmethod
  def _getFittingGraph(I, P):
    START_POINT_IDX = 0           # Index of rightmost Pareto point "s" at I = infinity
    LAST_POINT_IDX  = len(I) - 1  # Index of the last (i.e., leftmost) Pareto point
    graph = nx.DiGraph()          # The graph we're constructing
    hDict = {}                    # If a vertex v used a horizontal line to reach the "End" vertex,
                                  # hDict[v] gives the I-value of the line's endpoint
                                  
    # Note: All line slopes discussed in this function are technically <= 0. To make comparisons
    # (e.g., shallower/steeper, min/max) easier to follow, we use absolute values of these slopes
    # unless stated otherwise.

    # 1. Add the graph's main vertices. A vertex (x, y) exists if we can draw a line between
    #    points x and y without crossing below any other points.
    minSlope = np.zeros(len(I))
    for x in range(len(I)):
      for y in range(x + 1, len(I)):
        # Since I and P are Pareto points sorted right to left, y will always be left and above x
        # As we iterate over y's for an x, y will only move further left and up
        xPoint = (I[x], P[x])
        yPoint = (I[y], P[y])
        slope  = abs(Roofline._getSlope(xPoint, yPoint))

        # If the line xy is more shallow than any previous line from x,
        # xy travels below at least 1 Pareto point and can't be used for fitting
        if slope < minSlope[x]:
          continue
        minSlope[x] = slope

        # Compute xy's estimation error and add its vertex to the graph
        error = Roofline._getError(I, P, xPoint, yPoint)
        graph.add_node((x, y), slope=slope, error=error)

    # Each point's final minimum slope is recorded in the "minSlope" array. Any new line starting at
    # point p must have |slope| >= minSlope[p] to ensure it passes on or above all points left of p.

    # 2. Add the graph's main edges. An edge exists between vertices (x, y) and (y, z) if the line
    #    yz is steeper than the line xy. The edge's weight is yz's estimation error.
    for (x, y), xyData in graph.nodes(data=True):
      for (y2, z), yzData in graph.nodes(data=True):
        if y == y2 and yzData["slope"] >= xyData["slope"]:
          graph.add_edge((x, y), (y, z), weight=yzData["error"])

    # 3. Add the special "Start" vertex. This has an edge to all vertices (s, y) where s is the
    #    starting point at I = infinity. Each edge's weight is sy's estimation error.
    graph.add_node("Start")
    for vert, data in graph.nodes(data=True):
      if vert != "Start" and vert[0] == START_POINT_IDX:
        graph.add_edge("Start", vert, weight=data["error"])

    # 4. Add the special "End" vertex. Reaching this on a path from "Start" means that a valid
    #    piecewise fit has been constructed with segments defined by the vertices in the path.
    #    Edges between a vertex (x, y) and "End" fall into 3 categories
    #    1. If y is the leftmost Pareto point, a 0-weight edge is added to "End"
    #    2. If we can draw a line from y to the leftmost Pareto point without violating the fit
    #       rules (slopes right-to-left increase in steepness, all Pareto points are on or below
    #       the segments) do nothing. The segment that would correspond to adding an edge to "End"
    #       already exists in the graph. Thus, adding this edge would be redundant.
    #    3. If we can't draw a line from y to the leftmost Pareto point, we add a horizontal line
    #       to the point that y can reach. A edge is added to "End" weighted by the estimation
    #       error of the 2 resulting segments.
    graph.add_node("End")
    for vert, data in graph.nodes(data=True):
      if vert == "Start" or vert == "End":
        continue
      (x, y)  = vert
      xySlope = data["slope"]

      # If the line xy ends at the last Pareto point,
      if y == LAST_POINT_IDX:      
        # Add a 0-weight edge from (x, y) to "End". Since xy reaches the last
        # Pareto point, it's a valid way to complete the fitting process.
        graph.add_edge((x, y), "End", weight=0)
        continue

      yPoint  = (I[y], P[y])
      lPoint  = (I[LAST_POINT_IDX], P[LAST_POINT_IDX])
      ylSlope = abs(Roofline._getSlope(yPoint, lPoint))
      assert ylSlope != 0

      # A line between y and the last Pareto point "l" must have |slope|...
      # >= y's minimum |slope| - the line must travel above any points between y and l
      # >= xy's |slope|        - to stay concave-up, the line can't be shallower than xy
      ylSlopeMin = max(minSlope[y], xySlope)

      # If a line from y to l has a valid slope,
      if ylSlope >= ylSlopeMin:
        # The graph must contain an edge from (x, y) to (y, l). Since an edge
        # from (x, y) to "End" would be equivalent to adding a line from y to l
        # (an extra horizontal line isn't necessary), it would be redundant.
        assert graph.has_edge((x, y), (y, LAST_POINT_IDX))
        continue

      # At this point, l can't be reached with a direct line from y.
      # So, we'll add an extra horizontal line to l that y can reach.
      #
      # Examples with diagrams:
      # _________________________________________________________
      #      Line yl would go below z.  Adding hl lets y reach l.
      #
      #  l              |  l**            |  l*******h
      #                 |     **          |           *
      #            z    |       ***  z    |            z
      #                 |          **     |             *
      #              y  |            **y  |              y
      # _________________________________________________________
      #     Line xy is steeper than yl. Adding hl lets y reach l.
      #
      #  l              |  l***           |  l*******h
      #                 |      ***        |           *
      #            y    |         ***y    |            y
      #                 |             *   |             *
      #              x  |              x  |              x
      #

      # Compute the endpoint for the horizontal line
      # Point slope form:   P_l - P_y = slope * (I_y - I_h)
      # Solving for I_h:    I_h = (P_l - P_y) / slope + I_y
      I_h    = (P[LAST_POINT_IDX] - P[y]) / -ylSlopeMin + I[y]
      P_h    = P[LAST_POINT_IDX]
      hPoint = (I_h, P_h)

      # Compute the total error for the sloped and horizontal segments
      error  = Roofline._getError(I, P, yPoint, hPoint)
      error += Roofline._getError(I, P, hPoint, lPoint)

      # Add an edge to "End" weighted with the combined estimation errors
      # Record the I-value of the horizontal line's endpoint for later use
      graph.add_edge((x, y), "End", weight=error)
      hDict[(x, y)] = I_h

    return graph, hDict


  @staticmethod
  def _getSlope(point0, point1):
    (I_0, P_0) = point0
    (I_1, P_1) = point1
    return (P_1 - P_0) / (I_1 - I_0)


  @staticmethod
  def _getError(I, P, point0, point1):
    (I_0, P_0) = point0
    (I_1, P_1) = point1
    assert I_1 <  I_0, f"Assuming point1 is left of point0 (I_1 = {I_1}, I_0 = {I_0})"
    assert P_1 >= P_0, f"Assuming point1 is not below point0 (P_1 = {P_1}, P_0 = {P_0})"

    # Only computing error for points between the line's endpoints
    isBetween = (I_1 <= I) & (I <= I_0)

    # If the right endpoint is not at I = infinity,
    if not np.isposinf(I_0):
      # Use the line from point0 and point1 to estimate points between them
      slope     = Roofline._getSlope(point0, point1)  # Note: not using |slope|
      intercept = P_0 - slope * I_0
      P_est     = slope * I[isBetween] + intercept
    else:
      # Computing estimations w/ I_0 = infinity requires special handling.
      # We must estimate point0 and point1 with no error, but slope is 0.
      # The estimation becomes:
      #   P_0 when I = infinity
      #   P_1 when I < infinity
      assert np.isfinite(P_1) and np.isfinite(P_0) and np.isfinite(I_1), \
        "Assuming I_0 is the only non-finte argument"
      P_est = np.where(np.isposinf(I[isBetween]), P_0, P_1)

    # Return the sum of square errors for this linear estimation
    return np.sum((P[isBetween] - P_est)**2)


  @staticmethod
  def _getPathPoints(I, P, hDict, path):
    LAST_POINT_IDX = len(I) - 1   # Index of the last (i.e., leftmost) Pareto point

    I_path = np.empty(len(path) - 1)
    P_path = np.empty(len(path) - 1)

    # Collect segment starting points
    # I.e., (I[x], P[x]) for each vertex (x, y)
    path = np.array(path[1:-1])
    I_path[:-1] = I[path[:, 0]]
    P_path[:-1] = P[path[:, 0]]

    # Get the final point
    # I.e., (I[y], P[y]) from the final (x, y) vertex
    I_path[-1] = I[path[-1, 1]]
    P_path[-1] = P[path[-1, 1]]

    # If the path's last point isn't the last Pareto point,
    if path[-1, 1] != LAST_POINT_IDX:
      # We used a horizontal line to reach "End"
      # Add the horizontal line's endpoint to the path
      (x, y) = path[-1]
      I_h    = hDict[(x, y)]
      P_h    = P[LAST_POINT_IDX]
      I_path = np.append(I_path, I_h)
      P_path = np.append(P_path, P_h)

    return np.array(I_path), np.array(P_path)

# ----------------------------------------------------------------------------------------------- #
# Functions used by both classes
# ----------------------------------------------------------------------------------------------- #

def _convertToF64Array(X, errLabel):
  # Converts X to a numpy array with dtype=numpy.double (FP64)
  # If X is already this, X is returned without copying
  try:
    return np.asarray(X, dtype=np.double)
  except Exception as e:
    raise TypeError(f"Unable to convert {errLabel} to a numpy.ndarray of type numpy.double") from e


def _validateArrayDefault(X, infAllowed, errLabel):
  if np.isnan(X).any():
    raise ValueError(f"{errLabel} cannot be NaN")
  if (X < 0).any():
    raise ValueError(f"{errLabel} cannot be negative")
  if not infAllowed and np.isposinf(X).any():
    raise ValueError(f"{errLabel} cannot be infinity")