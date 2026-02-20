
Title

    wid -- Download data from WID.world

Syntax

        wid [, options]

    options                                             Description
    -------------------------------------------------------------------------
      indicators(list of 6-letter codes|_all)           codes names of the
                                                          indicators in the
                                                          database; default is
                                                          _all for all
                                                          indicators; see
                                                          options for details
      areas(list of area codes|_all)                    area code names of the
                                                          database; XX for
                                                          countries/regions,
                                                          XX-YY for
                                                          subregions; default
                                                          is _all for all
                                                          areas; see options
                                                          for details
      years(numlist)                                    years; default is all
      perc(list of percentiles|_all)                    list of percentiles;
                                                          either pXXpYY or
                                                          pXX; default is all_
                                                          for all percentiles;
                                                          see options for
                                                          details
      ages(list of age codes|_all)                      age category codes in
                                                          the database; 999
                                                          for all ages, 992
                                                          for adults; default
                                                          is _all for all age
                                                          categories; see
                                                          options for more
      population(list of population codes|_all)         type of population;
                                                          one-letter code, t
                                                          for tax units, i for
                                                          individuals; default
                                                          is _all for all
                                                          population types;
                                                          see options for more
      metadata                                          retrieve metadata (ie.
                                                          variable
                                                          descriptions,
                                                          sources,
                                                          methodological
                                                          notes, etc.)
      exclude                                           exclude interpolations
                                                          and extrapolations
                                                          from the results
      clear                                             replace data in memory
    -------------------------------------------------------------------------

Description

    wid imports data from the World Inequality Database (WID.world) directly
    into Stata.

Options

    indicators(list of 6-letter codes|_all) specify indicators to retrieve.
        Default is _all for all indicators.  You have to specify this option
        if you select all areas.  Indicators are 6-letter codes that
        corresponds to a given series type for a given income or wealth
        concept.  The first letter correspond to the type of series.  Some of
        the most common possibilities include:

              one-letter code      description
              -------------------------------------------------
              a                    average
              s                    share
              t                    threshold
              m                    macroeconomic total
              w                    wealth/income ratio
              -------------------------------------------------
              See wid.world/codes-dictionary (section "ONE-LETTER CODE FOR SERI
> ES TYPE")
              for the complete list.

        The next five letters correspond a concept (usually of income and
        wealth). Some of the most common possibilities include:

              five-letter code     description
              -------------------------------------------------
              ptinc                pre-tax national income
              pllin                pre-tax labor income
              pkkin                pre-tax capital income
              fiinc                fiscal income
              hweal                net personal wealth
              -------------------------------------------------
              See wid.world/codes-dictionary (section "FIVE-LETTER CODE FOR SER
> IES CONCEPT")
              for the complete list.

        For example, sfiinc corresponds to the share of fiscal income, ahweal
        corresponds to average personal wealth.

    areas(list of area codes|_all) specify areas for which to retrieve data.
        Default is _all for all areas.  You have to specify this option if
        you select all indicators.  Countries are coded using 2-letter ISO
        codes.  Country subregions are coded as XX-YY where XX is the country
        2-letter code.  Regions at PPP use custom 2-letter codes. Regions at
        market exchange rates use the same codes with the suffix -MER added.
        See wid.world/codes-dictionary (section "COUNTRY CODES") for the
        complete list of area codes.

    years(numlist) specify years for which to retrieve data. Default is all
        years.

    perc(list of percentiles|_all) specify which percentiles of the
        distribution to retrieve.  For share and average variables,
        percentiles correspond to percentile ranges and take the form pXXpYY.
        For example the top 1% share correspond to p99p100. The top 10% share
        excluding the top 1% is p90p99.  Thresholds associated to the
        percentile group pXXpYY correspond to the minimal income or wealth
        level that gets you into the group.  For example, the threshold of
        the percentile group p90p100 or p90p91 correspond to the 90%
        quantile.  Variables with no distributional meaning use the
        percentile p0p100.  See wid.world/codes-dictionary (section
        "PERCENTILE CODES") for more details.

    ages(list of age codes|_all) specify which age categories to retrieve.
        Ages are coded using 3-digit codes.  Some of the most common
        possibilities include:

              3-digit code         description
              -------------------------------------------------
              999                  all ages
              992                  adults, including elderly
                                     (20+)
              996                  adults, excluding elderly
                                     (20-65)
              -------------------------------------------------
              See wid.world/codes-dictionary (section "THREE-DIGIT CODE FOR AGE
>  GROUP")
              for the complete list.

    population(list of population codes|_all) specify which population
        categories to retrieve.  Population categories are coded using
        one-letter codes.  Some of the most common possibilities include:

              one-letter code      description
              -------------------------------------------------
              i                    individuals
              t                    tax units
              j                    equal-split adults (ie.
                                     income or wealth divided
                                     equally among spouses)
              -------------------------------------------------
              See wid.world/codes-dictionary (section "ONE-LETTER CODE FOR POPU
> LATION UNIT")
              for the complete list.

    metadata also retrieve metadata. Metadata provide, for each observation,
        the name and short description of the variable, of the age category,
        of the population category, the source of the data, and
        methodological notes.

    exclude exclude interpolation/extrapolations from the results. Some of
        the data on WID.world is the result of interpolations (when data is
        only available for a few years) or extrapolations (when data is not
        available for the most recent yea are based on much more limited
        information that other data points. We include these
        interpolations/extrapolation by default as a convenience, and also
        because these values are used to perform regional aggregations. Yet
        we stress that these estim pecially at the level of individual
        countries, can be fragile. For many purposes, it can be preferable to
        exclude these data points.

    clear replace data in memory, if any; if dataset is not empty and that
        option is not specified, the command will refuse to execute to avoid
        data losses.

Remarks

    Data is presented in long format (one observation per value).

    The complete and up-to-date description of the database is available
    online at wid.world/codes-dictionary.

    All monetary amounts are in local currency at constant prices for
    countries and country subregions.  Monetary amounts for world regions are
    in EUR PPP.  Series are at last year's prices, the database being usually
    updated every year in the summer.  To check the year of reference, look
    at when the price index inyixx is equal to 1.  You can access the price
    index using the indicator inyixx, the PPP exchange rates using xlcusp
    (USD), xlceup (EUR), xlcyup (CNY), and the market exchange rates using
    xlcusx (USD), xlceux (EUR), xlcyux (CNY).

    Shares and wealth/income ratios are given as a fraction of 1.  That is, a
    top 1% share of 20% is given as 0.2.  A wealth/income ratio of 300% is
    given as 3.

Examples

    The following examples only illustrate graphing, and do not leave any
    data in memory.

    Plot wealth inequality share in France:

        wid, indicators(shweal) areas(FR) perc(p90p100 p99p100) ages(992) pop(j
> ) clear

        // Reshape and plot
        reshape wide value, i(year) j(percentile) string
        label variable valuep90p100 "Top 10% share"
        label variable valuep99p100 "Top 1% share"

        graph twoway line value* year, title("Wealth inequality in France") ///
            ylabel(0.2 "20%" 0.4 "40%" 0.6 "60%" 0.8 "80%") ///
            subtitle("equal-split adults") ///
            note("Source: WID.world")

        (click to run)

    Plot the evolution of the pre-tax national income of the bottom 50% of
    the population in China, France and the United States since 1978 (in log
    scale):

        // Download and store the 2017 USD PPP exchange rate
        wid, indicators(xlcusp) areas(FR US CN) year(2017) clear
        rename value ppp
        tempfile ppp
        save "`ppp'"

        wid, indicators(aptinc) areas(FR US CN) perc(p0p50) year(1978/2017) age
> s(992) pop(j) clear
        merge n:1 country using "`ppp'", nogenerate

        // Convert to 2017 USD PPP (thousands)
        replace value = value/ppp/1000

        // Reshape and plot
        keep country year value
        reshape wide value, i(year) j(country) string
        label variable valueFR "France"
        label variable valueUS "United States"
        label variable valueCN "China"

        graph twoway line value* year, yscale(log) ylabel(1 2 5 10 20) ///
            ytitle("2017 PPP USD (000's)") ///
            title("Average pre-tax national income of the bottom 50%") subtitle
> ("equal-split adults") ///
            note("Source: WID.world") legend(rows(1))

        (click to run)

    Plot the long-run evolution of average net national income per adult in
    France, Germany, the United Kingdom and the United States (in log scale):

        // Download and store the 2017 USD PPP exchange rate
        wid, indicators(xlcusp) areas(FR US DE GB) year(2017) clear
        rename value ppp
        tempfile ppp
        save "`ppp'"

        // Download net national income in constant 2017 local currency
        wid, indicators(anninc) areas(FR US DE GB) age(992) clear
        merge n:1 country using "`ppp'", nogenerate

        // Convert to 2017 USD PPP (thousands)
        replace value = value/ppp/1000

        // Reshape and plot
        keep country year value
        reshape wide value, i(year) j(country) string
        label variable valueFR "France"
        label variable valueUS "United States"
        label variable valueDE "Germany"
        label variable valueGB "United Kingdom"

        graph twoway line value* year, yscale(log) ///
            ytitle("2017 PPP USD (000's)") ylabel(2 5 10 20 50 100) ///
            title("Average net national income") subtitle("per adult") ///
            note("Source: WID.world")

        (click to run)

Contact

    If you have comments, suggestions, or experience any problem with this
    command, please contact <thomas.blanchet@wid.world>.

