*******************************************************
* build_wid_percentiles.do
* Download WID percentile data for a single year,
* all available countries, then prepare a tidy file
* to export as CSV for use in Python/Colab.
*******************************************************

clear all
set more off

*------------------------------------------------------
* 0. Parameters — EDIT THESE
*------------------------------------------------------
local target_year = 2023     // <-- change to the year you want
local age_code    = 992      // adults (20+)
local pop_code j        // equal-split adults
local outdir      = "/Users/joehasell/Documents/OWID/notebooks/JoeHasell/WID_decomposition"      // folder where outputs will be saved
local perc_list ///
p0p1 p1p2 p2p3 p3p4 p4p5 p5p6 p6p7 p7p8 p8p9 p9p10 ///
p10p11 p11p12 p12p13 p13p14 p14p15 p15p16 p16p17 p17p18 p18p19 p19p20 ///
p20p21 p21p22 p22p23 p23p24 p24p25 p25p26 p26p27 p27p28 p28p29 p29p30 ///
p30p31 p31p32 p32p33 p33p34 p34p35 p35p36 p36p37 p37p38 p38p39 p39p40 ///
p40p41 p41p42 p42p43 p43p44 p44p45 p45p46 p46p47 p47p48 p48p49 p49p50 ///
p50p51 p51p52 p52p53 p53p54 p54p55 p55p56 p56p57 p57p58 p58p59 p59p60 ///
p60p61 p61p62 p62p63 p63p64 p64p65 p65p66 p66p67 p67p68 p68p69 p69p70 ///
p70p71 p71p72 p72p73 p73p74 p74p75 p75p76 p76p77 p77p78 p78p79 p79p80 ///
p80p81 p81p82 p82p83 p83p84 p84p85 p85p86 p86p87 p87p88 p88p89 p89p90 ///
p90p91 p91p92 p92p93 p93p94 p94p95 p95p96 p96p97 p97p98 p98p99 ///
p99p99.1 p99.1p99.2 p99.2p99.3 p99.3p99.4 p99.4p99.5 ///
p99.5p99.6 p99.6p99.7 p99.7p99.8 p99.8p99.9 p99.9p100

*------------------------------------------------------
* 1. Install wid command if needed
*------------------------------------------------------
cap which wid
if _rc {
    ssc install wid
}

*------------------------------------------------------
* 2. Grab the PPP conversion rates — LCUs per int-$ (xlcusp)
*------------------------------------------------------


wid, indicators(xlcusp) ///
    areas(_all)          ///
    years(`target_year') ///
    clear
	
rename value ppp
tempfile ppp
save "`ppp'"

* Keep relevant variables
keep country year ppp


* Export as CSV
export delimited using "`outdir'/ppps_`target_year'.csv", ///
    replace delim(",")



*------------------------------------------------------
* 2. Download average income and shares by percentile (aptinc sptinc)
*------------------------------------------------------

wid, indicators(aptinc sptinc) ///
    areas(_all)          ///
    years(`target_year') ///
    perc(`perc_list')    ///
    ages(`age_code')     ///
    population(`pop_code') ///
    clear

* Keep relevant variables
keep country year percentile variable value

* Strip the "992j" suffix so we have clean indicator names
gen indicator = substr(variable, 1, 6)   // "aptinc", "sptinc"

drop variable

* Wide reshape: one row per country-year-percentile, columns avg/share
reshape wide value, i(country year percentile) j(indicator) string

rename valueaptinc avg_lcu
rename valuesptinc share

* Convert to 2017 USD PPP
merge n:1 country using "`ppp'", nogenerate
gen double avg = avg_lcu/ppp


* Parse percentile code "pXXpYY" into numeric bounds (0-1)
* e.g. "p0p50" -> p_low=0.00, p_high=0.50
gen str10 p_clean = substr(percentile, 2, .)
split p_clean, parse("p") gen(p_)
destring p_1 p_2, replace
gen double p_low  = p_1 / 100
gen double p_high = p_2 / 100

drop p_clean p_1 p_2

* Export as CSV
export delimited using "`outdir'/wid_percentiles_`target_year'.csv", ///
    replace delim(",")


*------------------------------------------------------
* 3. Download total population and adult only population
*------------------------------------------------------
wid, indicators(npopul) ///
    areas(_all)          ///
    years(`target_year') ///
    ages(992 999)     ///
    population(i) ///
    clear

keep country year value variable

* Wide reshape: one row per country-year, columns total pop (999)/adult pop (992)
reshape wide value, i(country year) j(variable) string

rename valuenpopul992i adult_pop
rename valuenpopul999i total_pop

* Export as CSV
export delimited using "`outdir'/wid_population_`target_year'.csv", ///
    replace delim(",")

	
local target_year = 2023

local outdir      = "/Users/joehasell/Documents/OWID/notebooks/JoeHasell/WID_decomposition"      // folder where outputs will be saved

*------------------------------------------------------
* 4. Download aggregate net national income (mnninc)
*------------------------------------------------------
wid, indicators(mnninc) ///
    areas(_all)          ///
    years(`target_year') ///
    clear

keep country year value

rename value agg_net_national_income

* Export as CSV
export delimited using "`outdir'/wid_agg_income_`target_year'.csv", ///
    replace delim(",")
	
*******************************************************
* End of do-file
*******************************************************
