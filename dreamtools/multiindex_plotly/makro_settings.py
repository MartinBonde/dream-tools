DEFAULT_SET_AGGREGATIONS = {
	"a_": ["tot",],
	"a": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],

	"c_": ['cTot',],
	"c" : ['cBil', 'cEne', 'cVar', 'cBol', 'cTje', 'cTur'],

	"x_": ['xTot',],
	"x" : ['xEne', 'xVar', 'xSoe', 'xTje', 'xTur'],

	"g_": ['gTot',],
	"g" : ['g',],

	"s_": ['tot',],
	"sp": ['tje', 'fre', 'byg', 'soe', 'bol', 'lan', 'ene', 'udv'],
	"s" : ['tje', 'fre', 'byg', 'soe', 'bol', 'lan', 'ene', 'udv', 'off'],
	"r_": ['tot',],
	"r" : ['tje', 'fre', 'byg', 'soe', 'bol', 'lan', 'ene', 'udv', 'off'],

	"d_": ['tot',],

	"i_": ['iTot',],
	"i" : ['IM', 'IB', 'IL'],

	"k_": ['IM', 'IB'],
	"k" : ['IM', 'IB'],

    "portf_": ['NetFin',],
	"portf": ['NetFin', ],
}

LANGUAGE_LABELS = {
	"en": {
		"AGE_AXIS_TITLE": "age",
		"TIME_AXIS_TITLE": "",
		"YAXIS_TITLE_FROM_OPERATOR": {
			"pq": "Pct. change relative to baseline",
			"q": "Change relative to baseline",
			"m": "Difference from baseline",
			"pm": "Change in percentage points relative to baseline",
		},
	},
	"da": {
		"AGE_AXIS_TITLE": "Alder",
		"TIME_AXIS_TITLE": "",
		"YAXIS_TITLE_FROM_OPERATOR": {
			"pq": "Pct.-ændringer relativt til grundforløb",
			"q": "Ændring relativt til grundforløb",
			"m": "Forskel fra grundforløb",
			"pm": "Ændring i procentpoint relativt til grundforløb",
		},
	},
}

LANGUAGE = "en"

def set_language(language):
	global LANGUAGE
	LANGUAGE_LABELS[language]
	LANGUAGE = language

def age_axis_title():
	return LANGUAGE_LABELS[LANGUAGE]["AGE_AXIS_TITLE"]

def time_axis_title():
	return LANGUAGE_LABELS[LANGUAGE]["TIME_AXIS_TITLE"]

def yaxis_title_from_operator(operator):
	return LANGUAGE_LABELS[LANGUAGE]["YAXIS_TITLE_FROM_OPERATOR"].get(operator, "")

