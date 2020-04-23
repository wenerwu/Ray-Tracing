// Given a time between 0 and 1, evaluates a cubic polynomial with
// the given endpoint and tangent values at the beginning (0) and
// end (1) of the interval.  Optionally, one can request a derivative
// of the spline (0=no derivative, 1=first derivative, 2=2nd derivative).
template <class T>
inline T Spline<T>::cubicSplineUnitInterval(
    const T& position0, const T& position1, const T& tangent0,
    const T& tangent1, double normalizedTime, int derivative) {
  // TODO (Animation) Task 1a
  double t = normalizedTime;
  double t2 = t * t;
  double t3 = t2 * t;

  double h00 =  2 * t3 - 3 *  t2 + 1;
  double h10 = t3 - 2 * t2 + t;
  double h01 = -2 * t3 + 3 * t2;
  double h11 = t3 - t2;

  double h00_d = 6 * t2 - 6 * t;
  double h10_d = 3 * t2 - 4 * t;
  double h01_d = -6 * t2 + 6 * t;
  double h11_d = 3 * t2 - 2 * t;

  double h00_dd = 12 * t - 6;
  double h10_dd = 6 * t - 4;
  double h01_dd = -12 * t + 6;
  double h11_dd = 6 * t - 2;

  if(derivative == 0)
    return h00 * position0 + h10 * tangent0 + h01 * position1 + h11 * tangent1;
  else if(derivative == 1)
    return h00_d * position0 + h10_d * tangent0 + h01_d * position1 + h11_d * tangent1;
  else if(derivative == 2)
    return h00_dd * position0 + h10_dd * tangent0 + h01_dd * position1 + h11_dd * tangent1;


  printf("ERROR DERIVATIVE!\n");
  return T();
}

// Returns a state interpolated between the values directly before and after the
// given time.
template <class T>
inline T Spline<T>::evaluate(double time, int derivative) {
  // TODO (Animation) Task 1b
  // for(auto it = knots.begin(); it != knots.end(); it++)
  //   printf("%g ", it->first);
  // printf("\n");
  if (knots.size() < 1)
    return T();
  else if (knots.size() == 1)
  {
    if(derivative == 0)
      return knots.begin()->second;
    else 
      return T();
  }
  else if (time <= knots.begin()->first)
  {
    if(derivative == 0)
      return knots.begin()->second;
    else 
      return T();
  }
  else if (time >= knots.rbegin()->first)
  {
    if(derivative == 0)
      return knots.rbegin()->second;
    else 
      return T();
  }
  else
  {
    auto iter2 = knots.upper_bound(time);
    auto iter1 = iter2;
    iter1 --;
    double t0, t1, t2, t3;
    T p0, p1, p2, p3;
    t1 = iter1->first;
    t2 = iter2->first;

    p1 = iter1->second;
    p2  =iter2->second;
    if(iter1 == knots.begin())
    {
      // no iter0
      t0 = t1 - (t2 - t1);
      p0 = p1 - (p2 - p1);
    }
    else
    { 
      auto iter0 = iter1;
      iter0 --;
      t0 = iter0->first;
      p0 = iter0->second;
    }

    auto iter3 = iter2;
    iter3 ++;
    
    if(iter3 == knots.end())
    {
      // no iter3
      t3 = t2 + (t2 - t1);
      p3 = p2 + (p2 - p1);
    }
    else
    {
      t3 = iter3->first;
      p3 = iter3->second;
    }
   
    T m1 = (p2 - p0) / (t2 - t0);
    T m2 = (p3 - p1) / (t3 - t1);

    double normalizedTime = (time - t1) / (t2 - t1);
    return cubicSplineUnitInterval(p1, p2, m1, m2, normalizedTime, derivative);

   //  return T();
  }
  

}

// Removes the knot closest to the given time,
//    within the given tolerance..
// returns true iff a knot was removed.
template <class T>
inline bool Spline<T>::removeKnot(double time, double tolerance) {
  // Empty maps have no knots.
  if (knots.size() < 1) {
    return false;
  }

  // Look up the first element > or = to time.
  typename std::map<double, T>::iterator t2_iter = knots.lower_bound(time);
  typename std::map<double, T>::iterator t1_iter;
  t1_iter = t2_iter;
  t1_iter--;

  if (t2_iter == knots.end()) {
    t2_iter = t1_iter;
  }

  // Handle tolerance bounds,
  // because we are working with floating point numbers.
  double t1 = (*t1_iter).first;
  double t2 = (*t2_iter).first;

  double d1 = fabs(t1 - time);
  double d2 = fabs(t2 - time);

  if (d1 < tolerance && d1 < d2) {
    knots.erase(t1_iter);
    return true;
  }

  if (d2 < tolerance && d2 < d1) {
    knots.erase(t2_iter);
    return t2;
  }

  return false;
}

// Sets the value of the spline at a given time (i.e., knot),
// creating a new knot at this time if necessary.
template <class T>
inline void Spline<T>::setValue(double time, T value) {
  knots[time] = value;
}

template <class T>
inline T Spline<T>::operator()(double time) {
  return evaluate(time);
}
