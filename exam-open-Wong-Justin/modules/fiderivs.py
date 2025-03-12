import numpy as np
import pandas as pd




def price_bond(discounts: pd.DataFrame, cpnrate: float, ttm: float, cpnfreq: int = 2, face: float = 100) -> float:
    """
    Prices a typical coupon bond using discount factors.
    
    Parameters:
        discounts (pd.DataFrame): A dataframe with an index representing time-to-maturity (in years)
                                  at intervals (e.g., 0.5, 1.0, ..., 30), and with columns 'spot rate' and 'discount'.
        cpnrate (float): Annual coupon rate. If the value is greater than 1, it is assumed to be a percentage
                         and will be divided by 100.
        ttm (float): Time-to-maturity (in years) of the bond.
        cpnfreq (int, optional): Number of coupon payments per year (default=2 for semiannual coupons).
        face (float, optional): Face (par) value of the bond (default=100).
    
    Returns:
        float: The calculated bond price.
    """
    
    # Adjust coupon rate if provided as a percentage
    if cpnrate > 1:
        cpnrate = cpnrate / 100.0

    coupon_payment = face * cpnrate / cpnfreq

    # If time-to-maturity is less than one coupon period, set payment_dates to just [ttm]
    if ttm < 1/cpnfreq:
        payment_dates = np.array([ttm])
    else:
        # Generate regular coupon payment dates
        payment_dates = np.arange(1/cpnfreq, ttm + 1e-8, 1/cpnfreq)
        # If the last payment date is not exactly ttm, add the maturity as an irregular final period.
        if not np.isclose(payment_dates[-1], ttm):
            payment_dates = np.append(payment_dates, ttm)
    
    price = 0.0
    for t in payment_dates:
        # Determine cash flow:
        # For the final payment at maturity, add the face value.
        # Note: Depending on the bond's terms, you might want to prorate the coupon for an irregular period.
        if np.isclose(t, ttm):
            cash_flow = coupon_payment + face
        else:
            cash_flow = coupon_payment
        
        # Retrieve the discount factor: if an exact match is not found, interpolate.
        if t in discounts.index:
            discount_factor = discounts.loc[t, 'discount']
        else:
            discount_factor = np.interp(t, discounts.index.values, discounts['discount'].values)
        
        price += cash_flow * discount_factor
    
    return price






from dateutil.relativedelta import relativedelta

def compute_rates(discount_factors, ttm, compounding):
    """
    Computes annualized rates from given discount factors and time-to-maturity values.
    
    Parameters:
        discount_factors (array-like): Discount factors.
        ttm (array-like): Time-to-maturity values (in years).
        compounding (int or None): 
            - If an integer, indicates the number of compounding periods per year.
            - If None, continuous compounding is assumed.
    
    Returns:
        np.ndarray: An array of annualized rates (in decimals).
    
    Notes:
        For continuous compounding:
            discount_factor = exp(-rate * ttm)
            -> rate = -ln(discount_factor) / ttm
        
        For periodic compounding:
            discount_factor = (1 + rate/compounding)**(-compounding * ttm)
            -> rate = compounding * (discount_factor**(-1/(compounding * ttm)) - 1)
    """
    discount_factors = np.array(discount_factors)
    ttm = np.array(ttm)
    
    if compounding is None:
        # Continuous compounding
        rates = -np.log(discount_factors) / ttm
    else:
        # Periodic compounding
        rates = compounding * (discount_factors ** (-1/(compounding * ttm)) - 1)
    
    return rates









def compute_discount_factors(ttm, rates, compounding):
    """
    Computes discount factors for given ttm values and rates.
    
    Parameters:
        ttm (array-like): Time-to-maturity values (in years).
        rates (array-like): Corresponding annualized rates (in decimals).
        compounding (int or None): 
            - If an integer, indicates the number of compounding periods per year.
            - If None, continuous compounding is assumed.
    
    Returns:
        np.ndarray: An array of discount factors.
    """
    ttm = np.array(ttm)
    rates = np.array(rates)
    
    if compounding is None:
        # Continuous compounding
        discounts = np.exp(-rates * ttm)
    else:
        # Periodic compounding (e.g. 2 for semiannual, 4 for quarterly)
        discounts = (1 + rates / compounding) ** (-compounding * ttm)
    
    return discounts










def create_ttm_grid(anchor_ttm, interval_years=0.5, min_ttm=0.0, total_years=30.0):
    """
    Create a TTM grid from min_ttm to total_years (inclusive),
    ensuring that anchor_ttm is on the grid with step = interval_years.
    """
    # 1) Step backward from anchor_ttm down to min_ttm:
    #    anchor, anchor - step, anchor - 2*step, ... >= min_ttm
    left_points = []
    value = anchor_ttm
    while value >= min_ttm - 1e-12:  # small tolerance
        left_points.append(value)
        value -= interval_years
    left_points = list(sorted(set(left_points)))  # sort ascending; remove duplicates if any

    # 2) Step forward from anchor_ttm up to total_years:
    right_points = []
    value = anchor_ttm
    while value <= total_years + 1e-12:  # small tolerance
        right_points.append(value)
        value += interval_years
    right_points = list(sorted(set(right_points)))

    # 3) Merge them (they both include anchor_ttm):
    merged = sorted(set(left_points + right_points))

    # 4) Filter out anything slightly negative from rounding, or above total_years
    grid = [x for x in merged if x >= min_ttm - 1e-12 and x <= total_years + 1e-12]

    return np.array(grid)


def interpolate_curve(df,
                      interval_years=0.5,
                      total_years=30,
                      compounding=None,
                      anchor_ttm=0.5,
                      min_ttm=0.0):
    """
    Interpolates a spot rate curve derived from discount factors provided in df,
    ensuring that the new grid always includes anchor_ttm.

    The input DataFrame is expected to be indexed by time-to-maturity (ttm, in years),
    and contain columns 'date' and 'discount'.

    Steps:
      1. Convert discount factors to spot rates (spot_rates).
      2. Create a TTM grid that includes `anchor_ttm` in uniform steps of `interval_years`
         from `min_ttm` up to `total_years`.
      3. Linearly interpolate spot_rates onto this new grid.
      4. Compute new discount factors from these interpolated spot rates.
      5. Compute "maturity date" for each new TTM by shifting a "quote_date" from the earliest input TTM.

    Returns:
        A DataFrame indexed by ttm with columns: 'maturity date', 'spot rate', 'discount'.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Sort and drop duplicates on TTM.
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]

    # Derive quote_date from the row with the smallest ttm: date = quote_date + ttm * 365.25
    min_ttm_in_data = df.index.min()
    corresponding_date = df.loc[min_ttm_in_data, 'date']
    quote_date = (corresponding_date - pd.to_timedelta(min_ttm_in_data * 365.25, unit='D')).date()

    # -- Example function to compute spot rates from discount factors (you supply this) --
    spot_rates = compute_rates(df['discount'].values, df.index.values, compounding)

    # --- Create the new TTM grid that definitely includes anchor_ttm ---
    new_ttm = create_ttm_grid(anchor_ttm=anchor_ttm,
                              interval_years=interval_years,
                              min_ttm=min_ttm,
                              total_years=total_years)

    # -- Interpolate spot rates onto new TTM grid --
    interpolated_spot = np.interp(new_ttm, df.index.values, spot_rates)

    # Build a new DataFrame
    curve = pd.DataFrame({'spot rate': interpolated_spot}, index=new_ttm)
    curve.index.name = 'ttm'

    # Compute new discount factors from the interpolated spot rates (you supply this as well)
    new_discounts = compute_discount_factors(curve.index.values,
                                             curve['spot rate'].values,
                                             compounding)
    curve['discount'] = new_discounts

    # Example approach for computing "maturity date" using calendar arithmetic for half-year intervals
    def compute_date(ttm):
        if abs(interval_years - 0.5) < 1e-12:
            # use half-year increments with relativedelta
            whole_years = int(ttm)
            remainder = ttm - whole_years
            if abs(remainder - 0.5) < 1e-12:
                return quote_date + relativedelta(years=whole_years, months=6)
            else:
                return quote_date + relativedelta(years=whole_years)
        else:
            # fallback: approximate with 365.25 days per year
            return (pd.Timestamp(quote_date) +
                    pd.to_timedelta(ttm * 365.25, unit='D')).date()

    curve['maturity date'] = [compute_date(x) for x in curve.index]

    # Reorder columns
    curve = curve[['maturity date', 'spot rate', 'discount']]

    return curve






def discounts_to_forwardcurve(discountcurve, n_compound=None, dt=None):
    """
    Convert discount factors into forward rates.

    Parameters
    ----------
    discountcurve : pd.Series or pd.DataFrame
        Discount factors indexed by time (e.g. year-fractions).
    n_compound : int, optional
        Compounding frequency per year.  If None, use continuous compounding.
    dt : float, optional
        Time step (fraction of a year) between index points.  
        If None, inferred as index[1] - index[0].
    """

    # If user doesn't pass dt, try a uniform spacing from the index
    if dt is None:
        dt = discountcurve.index[1] - discountcurve.index[0]
        
    # Ratio of consecutive discount factors
    # This is the discount factor for that sub-interval
    F = discountcurve / discountcurve.shift(1)

    if n_compound is None:
        # (A) Use continuous compounding
        # Forward rate f_i = - (1/dt) * ln(F_i)
        forwardcurve = - np.log(F) / dt
    else:
        # (B) Use nominal rate with n_compound compounding
        # forward = n_compound * ( (1/F)^(1/(n_compound*dt)) - 1 )
        forwardcurve = n_compound * ((1.0 / F) ** (1.0 / (n_compound * dt)) - 1.0)

    return forwardcurve





def duration_closed_formula(tau, ytm, cpnrate=None, freq=2):

    if cpnrate is None:
        cpnrate = ytm
        
    y = ytm/freq
    c = cpnrate/freq
    T = tau * freq
        
    if cpnrate==ytm:
        duration = (1+y)/y  * (1 - 1/(1+y)**T)
        
    else:
        duration = (1+y)/y - (1+y+T*(c-y)) / (c*((1+y)**T-1)+y)

    duration /= freq
    
    return duration




    
def price_bond_textbook(ytm, T, cpn, cpnfreq=2, face=100, accr_frac=None):
    ytm_n = ytm/cpnfreq
    cpn_n = cpn/cpnfreq
    
    if accr_frac is None:
        #accr_frac = 1 - (T-round(T))*cpnfreq        
        accr_frac = 0

    if cpn==0:
        accr_frac = 0
        
    N = T * cpnfreq
    price = face * ((cpn_n / ytm_n) * (1-(1+ytm_n)**(-N)) + (1+ytm_n)**(-N)) * (1+ytm_n)**(accr_frac)
    return price





from scipy.optimize import fsolve

def ytm(price, T, cpn, cpnfreq=2, face=100, accr_frac=None,solver='fsolve',x0=.01):
    
    pv_wrapper = lambda y: price - price_bond_textbook(y, T, cpn, cpnfreq=cpnfreq, face=face, accr_frac=accr_frac)

    if solver == 'fsolve':
        ytm = fsolve(pv_wrapper,x0)
    elif solver == 'root':
        ytm = root(pv_wrapper,x0)
    return ytm




import numpy as np

def price_bond_dirty(ytm,
                     T,
                     cpn,
                     cpnfreq=2,
                     face=100,
                     accr_frac=0.0):
    """
    Compute the 'dirty' price of a standard (vanilla) coupon bond,
    allowing for settlement between coupon dates.

    Parameters
    ----------
    ytm : float
        Annual yield (nominal), compounded 'cpnfreq' times per year.
        E.g. 0.04 means 4.0% nominal annual yield.
    T : float
        Total time (in years) from settlement to maturity.
        (Assumed to be an integer multiple of 1/cpnfreq if we want exactly N coupons.)
    cpn : float
        Annual coupon rate (as a decimal); e.g. 0.044 = 4.4% per year.
    cpnfreq : int
        Number of coupon payments per year, e.g. 2 for semiannual.
    face : float
        Bond face value (e.g. 100).
    accr_frac : float
        Fraction of the current coupon period that has ALREADY elapsed
        since the last coupon. 
        - 0.0 => we are exactly on a coupon date.
        - 0.5 => half the coupon period has passed.
        - 0.99 => we are 99% of the way to the next coupon date, etc.

    Returns
    -------
    dirty_price : float
        The full (dirty) price of the bond (per 100 face if face=100).
    """

    # Per-period yield
    i = ytm / cpnfreq

    # Per-coupon payment in dollars
    c = face * (cpn / cpnfreq)

    # Total number of (remaining) coupon payments (assuming T * cpnfreq is an integer)
    N = int(np.round(T * cpnfreq))

    # --- Usual textbook bond price as if settlement is exactly on a coupon date ---
    #    P_0 = sum of coupon annuity + redemption discounted N full periods
    #    (face factor pulled out for convenience)
    # -----------------------------------------------------------------------------
    pv_factor = (1 + i) ** (-N)
    annuity_factor = (1 - pv_factor) / i
    price_on_coupon_date = c * annuity_factor + face * pv_factor
    # That is the price at a coupon date. Now we must adjust for settlement
    # that occurs accr_frac into the coupon period.

    # --- Adjust discounting if we are between coupons ---
    # The next coupon is only (1 - accr_frac) periods away instead of 1 full period.
    # => multiply by (1 + i)^(accr_frac)
    #    (this shortens the discount on each cash flow).
    dirty_price = price_on_coupon_date * ((1 + i) ** accr_frac)

    return dirty_price




def price_bond_clean(ytm,
                     T,
                     cpn,
                     cpnfreq=2,
                     face=100,
                     accr_frac=0.0):
    """Compute the market 'clean' price = dirty price - accrued interest."""
    dirty = price_bond_dirty(ytm, T, cpn, cpnfreq, face, accr_frac)
    coupon_per_period = face * (cpn / cpnfreq)
    AI = coupon_per_period * accr_frac
    return dirty - AI



from scipy.optimize import fsolve

def solve_ytm(bond_price,
              T,
              cpn,
              cpnfreq=2,
              face=100,
              accr_frac=0.0,
              guess=0.03,
              clean=True):
    """
    Solve for the yield to maturity that reproduces `bond_price` given
    the bond parameters and the fraction of the coupon period elapsed.

    Parameters
    ----------
    bond_price : float
        Observed price (clean or dirty).
    T : float
        Years from settlement to maturity.
    cpn : float
        Annual coupon rate (decimal).
    cpnfreq : int
        Number of coupons per year.
    face : float
        Face value (e.g. 100).
    accr_frac : float
        Fraction of the current coupon period elapsed.
    guess : float
        Initial guess for YTM (annual).
    clean : bool
        If True, interpret `bond_price` as a clean price and solve for YTM.
        If False, interpret it as the dirty price.

    Returns
    -------
    ytm_solution : float
        The yield to maturity (annual, nominal, compounding 'cpnfreq' times).
    """
    
    def objective(y):
        if clean:
            return price_bond_clean(y, T, cpn, cpnfreq, face, accr_frac) - bond_price
        else:
            return price_bond_dirty(y, T, cpn, cpnfreq, face, accr_frac) - bond_price

    ytm_sol = fsolve(objective, guess)
    return ytm_sol[0]









def discount_factors_to_rates(
    df_series: pd.Series,
    freq: int | None = None
) -> pd.Series:
    """
    Convert a Series of discount factors (indexed by time-to-payoff in years)
    into annualized discount rates.

    Parameters
    ----------
    df_series : pd.Series
        The discount factors, indexed by time to payoff (in years).
        These should be between 0 and 1, and times should be > 0.
    freq : int or None
        If None (default), use continuous compounding.
        If an integer (e.g. 2, 4, 12), use discrete compounding that many times per year.

    Returns
    -------
    pd.Series
        Annualized discount rates, with the same index.
    """
    t_array = df_series.index.to_numpy()
    df_array = df_series.to_numpy()

    rates = np.zeros_like(df_array, dtype=float)

    if freq is None:
        # Continuous compounding: r = -ln(DF) / t
        rates = -np.log(df_array) / t_array
    else:
        # Discrete compounding: r = m * ((1 / DF)^(1 / (m*t)) - 1)
        m = float(freq)
        with np.errstate(divide='ignore', invalid='ignore'):
            rates = m * ((1.0 / df_array) ** (1.0 / (m * t_array)) - 1.0)

    return pd.Series(data=rates, index=df_series.index, name="discount_rates")








def rates_to_discount_factors(
    rate_series: pd.Series,
    freq: int | None = None
) -> pd.Series:
    """
    Convert a Series of annualized discount rates (indexed by time-to-payoff in years)
    into discount factors.

    Parameters
    ----------
    rate_series : pd.Series
        The discount rates, indexed by time to payoff (in years).
        Times should be > 0.
    freq : int or None
        If None (default), use continuous compounding.
        If an integer (e.g. 2, 4, 12), use discrete compounding that many times per year.

    Returns
    -------
    pd.Series
        Discount factors, with the same index.
    """
    t_array = rate_series.index.to_numpy()
    r_array = rate_series.to_numpy()

    df_array = np.zeros_like(r_array, dtype=float)

    if freq is None:
        # Continuous compounding: DF = exp(-r * t)
        df_array = np.exp(-r_array * t_array)
    else:
        # Discrete compounding: DF = (1 + r/m)^(-m * t)
        m = float(freq)
        df_array = (1.0 + r_array / m) ** (-m * t_array)

    return pd.Series(data=df_array, index=rate_series.index, name="discount_factors")








import math
from scipy.stats import norm

def black_option_price(forward, strike, implied_vol, time_to_expiry, discount_factor, option_type="call"):
    """
    Prices an option on an interest rate derivative using Black's formula.
    
    Parameters:
        forward (float): The forward price.
        strike (float): The strike price.
        implied_vol (float): The annualized implied volatility.
        time_to_expiry (float): Time to expiration in years.
        discount_factor (float): The discount factor to expiry.
        option_type (str): "call" or "put". Default is "call".
        
    Returns:
        float: The option price.
    """
    # Avoid division by zero if time_to_expiry is 0.
    if time_to_expiry <= 0 or implied_vol <= 0:
        # Option is exercised immediately (or volatility is zero): 
        # Return intrinsic value discounted.
        if option_type.lower() == "call":
            return discount_factor * max(forward - strike, 0)
        else:
            return discount_factor * max(strike - forward, 0)
    
    sigma_sqrt_t = implied_vol * math.sqrt(time_to_expiry)
    d1 = (math.log(forward / strike) + 0.5 * implied_vol**2 * time_to_expiry) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    
    if option_type.lower() == "call":
        price = discount_factor * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    elif option_type.lower() == "put":
        price = discount_factor * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return price






def blacks_formula(T,vol,strike,fwd,discount=1,isCall=True):
        
    sigT = vol * np.sqrt(T)
    d1 = (1/sigT) * np.log(fwd/strike) + .5*sigT
    d2 = d1-sigT
    
    if isCall:
        val = discount * (fwd * norm.cdf(d1) - strike * norm.cdf(d2))
    else:
        val = discount * (strike * norm.cdf(-d2) - fwd * norm.cdf(-d1))
    return val







def highlight(x,idx,col,highlight_style="background-color: teal"):
    # Create a DataFrame of empty strings with the same shape as x
    df_styles = pd.DataFrame("", index=x.index, columns=x.columns)
    # Set the style for the desired cell(s). Here row index 2 and column 'C'
    df_styles.loc[idx, col] = highlight_style
    return df_styles





def get_approximate_discount(T,discs):
    diffs_array = np.abs(discs.index - T)
    imin = diffs_array.argmin()
    idx = discs.index[imin] 
    return idx




def calc_forward_bond_price(spot,Tfwd,discount_curve,cpnrate,face=100,cpnfreq=2):
    
    discount_grid_step = np.diff(discount_curve.index).mean()
    grid_step_cpn = round(1 / (cpnfreq * discount_grid_step))
    Tfwd_rounded = get_approximate_discount(Tfwd,discount_curve)

    Z = discount_curve.loc[Tfwd_rounded,'discount']
    cpn_discs = discount_curve.loc[:Tfwd_rounded:grid_step_cpn,'discount']

    coupon_payment = face * cpnrate / cpnfreq
    pv_coupons = sum(coupon_payment * df for df in cpn_discs)
    fwd_price = (spot - pv_coupons) / Z

    return fwd_price






def ratevol_to_pricevol(ratevol,rate,duration):
    pricevol = ratevol * rate * duration
    return pricevol



from scipy.stats import norm

def normal_cdf(x):
    return(1 + math.erf(x/np.sqrt(2)))/2

def normal_pdf(x):
    return np.exp(-x**2/2) / np.sqrt(2*np.pi)

def bs_normargs(under=None,strike=None,T=None,rf=None,vol=None):
    d1 = (np.log(under/strike) + (rf + .5 * vol**2)*T ) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return [d1,d2]

def bs_delta(under=None,strike=None,T=None,rf=None,vol=None):
    d1 = bs_normargs(under=under,strike=strike,T=T,rf=rf,vol=vol)[0]

    return normal_cdf(d1)         











def price_callable_bond(discounts,ttm,tte,cpnrate,ivol,strike=None,accint=0,face=100,recalculate_ivol=False,rateshock=0):

    discs = rates_to_discount_factors(discount_factors_to_rates(discounts['discount']) + rateshock).rename('discount').to_frame()
    Topt_rounded = get_approximate_discount(tte,discs)
    
    Pspot = price_bond(discs,cpnrate=cpnrate,ttm=ttm)
    Pspot += accint
    Pfwd = calc_forward_bond_price(Pspot,tte,discs,cpnrate,face=face)

    if strike is None:
        strike = face + accint

    if recalculate_ivol:        
        frates = discounts_to_forwardcurve(discs['discount'],n_compound=None)
        fwdrate = frates.loc[Topt_rounded] 
        y = ytm(Pspot, ttm, cpnrate)[0]
        duration = duration_closed_formula(ttm,y,cpnrate = cpnrate)
        ivol_price = ratevol_to_pricevol(ivol,fwdrate,duration)
    
    else:
        ivol_price = ivol

    Z = discs.loc[Topt_rounded,'discount']
    value_call = black_option_price(Pfwd, strike=strike, implied_vol=ivol_price, time_to_expiry=tte, discount_factor=Z, option_type="call")
    P_callable = Pspot - value_call

    return P_callable