*******************************************************
* fetch_population.do
* Download population data from WID
* Fast - no percentile data
*******************************************************

clear all
set more off

local target_year = 2023
local outdir = "/Users/joehasell/Documents/GitHub/data_work/global_inequality_pip_wid/inputs"

*------------------------------------------------------
* Download total and adult population (npopul)
*------------------------------------------------------

wid, indicators(npopul) ///
    areas(_all)          ///
    years(`target_year') ///
    ages(992 999)        ///
    population(i)        ///
    clear

keep country year value variable

* Reshape wide: one row per country-year
* adult pop (992) and total pop (999)
reshape wide value, i(country year) j(variable) string

rename valuenpopul992i adult_pop
rename valuenpopul999i total_pop

* Export as CSV
export delimited using "`outdir'/WID_aggregate_population.csv", ///
    replace delim(",")

*******************************************************
* End of do-file
*******************************************************
