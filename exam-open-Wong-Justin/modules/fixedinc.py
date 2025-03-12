import numpy as np
#import datetime
import holidays
import pandas as pd
from pandas.tseries.offsets import BDay
from datetime import datetime








def get_coupon_dates(quote_date, maturity_date):
    """
    Calculate semiannual coupon payment dates for a bond.

    Parameters:
    - quote_date (str or datetime): The quote date of the bond.
    - maturity_date (str or datetime): The maturity date of the bond.

    Returns:
    - list of datetime: Semiannual coupon payment dates after the quote date.
    """
    # Convert input to datetime if needed
    if isinstance(quote_date, str):
        quote_date = datetime.strptime(quote_date, '%Y-%m-%d')
    if isinstance(maturity_date, str):
        maturity_date = datetime.strptime(maturity_date, '%Y-%m-%d')
    
    # Ensure dates are valid
    if quote_date >= maturity_date:
        raise ValueError("Quote date must be earlier than maturity date.")

    # Generate coupon dates
    semiannual_offset = pd.DateOffset(months=6)
    dates = []
    current_date = maturity_date

    while current_date > quote_date:
        dates.append(current_date)
        current_date -= semiannual_offset

    # Return sorted dates in ascending order
    return sorted(dates)








def calc_cashflows(quote_data, filter_maturity_dates=False):
    # Detect the file type and map column names
    if 'CALDT' in quote_data.columns:
        date_col = 'CALDT'
        maturity_col = 'TMATDT'
        coupon_col = 'TCOUPRT'
    elif 'quote date' in quote_data.columns:
        date_col = 'quote date'
        maturity_col = 'maturity date'
        coupon_col = 'cpn rate'
    else:
        raise ValueError("Unrecognized data format: Missing key columns.")

    # Initialize the cashflow matrix
    CF = pd.DataFrame(dtype=float, data=0, index=quote_data.index, columns=quote_data[maturity_col].unique())

    for i in quote_data.index:
        # Get coupon dates
        coupon_dates = get_coupon_dates(quote_data.loc[i, date_col], quote_data.loc[i, maturity_col])

        if coupon_dates is not None:
            # Add coupon payments
            CF.loc[i, coupon_dates] = quote_data.loc[i, coupon_col] / 2

        # Add face value at maturity
        CF.loc[i, quote_data.loc[i, maturity_col]] += 100

    # Clean up the cashflow matrix
    CF = CF.fillna(0).sort_index(axis=1)
    CF.drop(columns=CF.columns[(CF == 0).all()], inplace=True)

    if filter_maturity_dates:
        CF = filter_treasury_cashflows(CF, filter_maturity_dates=True)

    return CF









def filter_treasuries(data, t_date=None, filter_maturity=None, filter_maturity_min=None, drop_duplicate_maturities=False, filter_tips=True, filter_yld=True):
    outdata = data.copy()

    # Detect the file type and map column names
    if 'CALDT' in outdata.columns:
        date_col = 'CALDT'
        maturity_col = 'TMATDT'
        type_col = 'ITYPE'
        yield_col = 'TDYLD'
    elif 'quote date' in outdata.columns:
        date_col = 'quote date'
        maturity_col = 'maturity date'
        type_col = 'type'
        yield_col = 'ytm'
    else:
        raise ValueError("Unrecognized data format: Missing key columns.")

    if t_date is None:
        t_date = outdata[date_col].values[-1]

    outdata = outdata[outdata[date_col] == t_date]

    # Filter out redundant maturity
    if drop_duplicate_maturities:
        outdata = outdata.drop_duplicates(subset=[maturity_col])

    # Filter by max maturity
    if filter_maturity is not None:
        mask_truncate = outdata[maturity_col] < (t_date + np.timedelta64(365 * filter_maturity + 1, 'D'))
        outdata = outdata[mask_truncate]

    # Filter by min maturity
    if filter_maturity_min is not None:
        mask_truncate = outdata[maturity_col] > (t_date + np.timedelta64(365 * filter_maturity_min - 1, 'D'))
        outdata = outdata[mask_truncate]

    outdata = outdata[outdata[type_col].isin([11, 12]) == (not filter_tips)]

    if filter_yld:
        outdata = outdata[outdata[yield_col] > 0]

    return outdata








def filter_treasury_cashflows(CF, filter_maturity_dates=False, filter_benchmark_dates=False, filter_CF_strict=True):

    mask_benchmark_dts = []

    # Filter by using only benchmark treasury dates
    for col in CF.columns:
        if filter_benchmark_dates:
            if col.month in [2, 5, 8, 11] and col.day == 15:
                mask_benchmark_dts.append(col)
        else:
            mask_benchmark_dts.append(col)

    if filter_maturity_dates:
        mask_maturity_dts = CF.columns[(CF >= 100).any()]
    else:
        mask_maturity_dts = CF.columns

    mask = [i for i in mask_benchmark_dts if i in mask_maturity_dts]

    CF_filtered = CF[mask]

    if filter_CF_strict:
        # Drop issues that had CF on excluded dates
        mask_bnds = CF_filtered.sum(axis=1) == CF.sum(axis=1)
        CF_filtered = CF_filtered[mask_bnds]

    else:
        # Drop issues that have no CF on included dates
        mask_bnds = CF_filtered.sum(axis=1) > 0
        CF_filtered = CF_filtered[mask_bnds]

    # Update to drop dates with no CF
    CF_filtered = CF_filtered.loc[:, (CF_filtered > 0).any()]

    return CF_filtered


















import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

def heatmap_year_month(s, plottitle='',figsize=(8,8)):
    """
    s: A pandas Series with a DateTimeIndex, exactly one value per year-month.
       The day component is assumed to be the last day of the month.
    plottitle: Optional title for the heatmap.
    """

    # Convert your Series into a DataFrame so we can add 'year' and 'month' columns
    df = s.to_frame(name='values')
    
    # Extract year and month from the DateTimeIndex
    df['year'] = df.index.year
    df['month'] = df.index.month

    # Pivot so that each row is a year, each column is a month, and the cell is the value
    pivoted = df.pivot(index='year', columns='month', values='values')

    # Ensure columns go from 1 to 12, even if some months are missing
    pivoted = pivoted.reindex(columns=range(1, 13))

    # Create month labels (Jan, Feb, ..., Dec)
    month_abbr = [calendar.month_abbr[m] for m in pivoted.columns]

    # Now plot the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        pivoted, 
        xticklabels=month_abbr,        # Show 'Jan', 'Feb', ...
        yticklabels=pivoted.index,     # Show actual year
        #cmap="viridis",                # or any other colormap
        #annot=True,                    # optional: show values in each cell
        fmt=".2f",                     # format for the numbers
        cbar=True
    )

    plt.xlabel('Month')
    plt.ylabel('Year')
    plt.title(plottitle)
    plt.show()

















import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

def heatmap_weekly(
    s: pd.Series,
    plottitle: str = "",
    cmap: str = "viridis",
    annot: bool = False,
    fmt: str = ".2f"
):
    """
    Plots a heatmap of weekly data with year on the y-axis and week-of-year on the x-axis.
    Only places month labels on the x-axis when crossing into a new month.

    Parameters
    ----------
    s : pd.Series
        A pandas Series with a DateTimeIndex at weekly frequency.
        For multiple years, the Series can span multiple years.
    plottitle : str
        Optional title for the plot.
    cmap : str
        Colormap for the heatmap (default 'viridis').
    annot : bool
        Whether to display numerical values in each cell (default False).
    fmt : str
        String format for annotations (default '.2f').
    """

    # 1. Convert Series to DataFrame to add columns for year & week
    df = s.to_frame(name="values")
    
    # Create columns for year and ISO week number
    # (ISO weeks run Monday-Sunday, with possible week 53)
    df["year"] = df.index.isocalendar().year
    df["week"] = df.index.isocalendar().week  # 1..52 or 53
    
    # 2. Pivot to get rows = year, columns = week
    pivoted = df.pivot(index="year", columns="week", values="values")
    # pivoted.index  = unique years
    # pivoted.columns = 1..52 (or 53), depending on data
    
    # 3. Build a map of (year, week) -> earliest date in that group
    #    We'll group by (year, week) and pick the min date from the index.
    #    That date is used to detect month boundaries.
    df["actual_date"] = df.index  # keep the actual timestamps in a column
    group_min_date = df.groupby(["year", "week"])["actual_date"].min()
    # group_min_date is a Series with MultiIndex (year, week)
    
    # 4. Build custom x tick labels based on month boundaries.
    #
    # We need one label per "week" column. But for each column (e.g. week=5),
    # we might have multiple years. We'll pick the earliest year in pivoted.index
    # to get a "representative date." This ensures a single row of labels works
    # across multiple years.
    #
    # If the user wants a different labeling approach (e.g. separate labels
    # for each row/year), that requires a more complex approach (facet grids or
    # multiple subplots). This code keeps it simple with one set of labels.
    
    all_weeks = pivoted.columns  # e.g. [1, 2, 3, ..., 52]
    if len(all_weeks) == 0:
        raise ValueError("No weekly columns found. Is your Series empty?")
    
    earliest_year = pivoted.index.min()  # the smallest year in your data
    xlabels = []
    prev_month = None

    for w in all_weeks:
        if (earliest_year, w) in group_min_date.index:
            # Get the earliest date for that (year, week)
            date_for_week = group_min_date.loc[(earliest_year, w)]
            current_month = date_for_week.month
            # If the month changed from previous column, label it
            if current_month != prev_month:
                xlabels.append(calendar.month_abbr[current_month])
                prev_month = current_month
            else:
                xlabels.append("")  # same month as previous column -> blank
        else:
            # If the earliest_year doesn't have data for this week,
            # label it blank
            xlabels.append("")
    
    # 5. Plot the heatmap
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(
        pivoted,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        xticklabels=xlabels,        # custom monthly boundary labels
        yticklabels=pivoted.index,  # the actual years
        cbar=True
    )
    ax.set_xlabel("Week of Year (labeled by month boundary)")
    ax.set_ylabel("Year")
    ax.set_title(plottitle if plottitle else "Weekly Heatmap")
    plt.tight_layout()
    plt.show()









import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def heatmap_daily_no_xticks(s, plottitle=""):
    """
    Plots a heatmap of daily data with:
      - Rows = year
      - Columns = day_of_year (1..365 or 366)
    NO x-axis ticks or labels are shown.

    Parameters
    ----------
    s : pd.Series
        A pandas Series with a DateTimeIndex at daily frequency (or at least 
        covering distinct days).
    plottitle : str
        Optional title for the plot.
    """

    # 1) Convert the Series to a DataFrame to extract year and day_of_year
    df = s.to_frame(name="values")
    df["year"] = df.index.year
    df["day_of_year"] = df.index.dayofyear

    # 2) Pivot so each row is a year, each column is a day_of_year
    pivoted = df.pivot(index="year", columns="day_of_year", values="values")

    if pivoted.empty:
        raise ValueError("No data found. Is your Series empty or invalid?")

    # 3) Plot the heatmap
    plt.figure(figsize=(12, 5))
    ax = sns.heatmap(
        pivoted,
        cbar=True,
        xticklabels=False,  # Disable x tick labels
        yticklabels=True    # Keep year labels on y-axis
    )
    
    # 4) Remove the x-axis ticks entirely
    ax.set_xticks([])  # Remove tick marks
    ax.set_xlabel("")   # Remove any default x-axis label
    ax.set_ylabel("Year")
    ax.set_title(plottitle)

    plt.tight_layout()
    plt.show()




