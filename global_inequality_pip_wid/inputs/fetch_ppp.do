*******************************************************
* fetch_ppp.do
* Download PPP conversion factors from WID
* Fast - no percentile data
*******************************************************

clear all
set more off

local target_year = 2023
local outdir = "/Users/joehasell/Documents/GitHub/data_work/global_inequality_pip_wid/inputs"

*------------------------------------------------------
* Grab PPP conversion rates (xlcusp)
*------------------------------------------------------

wid, indicators(xlcusp) ///
    areas(_all)          ///
    years(`target_year') ///
    clear

rename value ppp

* Keep relevant variables
keep country year ppp

* Add percentile column (needed for Python code filtering)
gen percentile = "p0p100"

* Export as CSV
export delimited using "`outdir'/WID_ppp.csv", ///
    replace delim(",")

*******************************************************
* End of do-file
*******************************************************
