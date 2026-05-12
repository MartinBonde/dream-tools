DEFAULT_SET_AGGREGATIONS = {

	"c_": ['cTot',],
	"c" : ['cHouEne','cHou','cCarEne','cCar','cFoodDairy','cFoodVeg','cFoodBev','cFoodPig','cFoodCow','cFoodFish','cFoodPoul','cNonFood','cSer','cTou'],
	
  "Ani_sectors_":['tot'],
  "Ani_sectors":['01031','01032','01051','01052','01061','01062','01070'],
	
  "Plant_sectors_":['tot'],
	"Plant_sectors":['01011','01012','01020'],
	
  "land5":['wetland','settlement','forest','sea','crop','grass'],
	
  "liabilities":['Mortgages','Debt'],

	"x_": ['xTot',],
	"x" : ['xOth', 'xTur'],

	"g_": ['gTot',],
	"g" : ['g',],

	"i_": ['iTot',],
	"i" : ['iM', 'iB', 'iT','invt'],

	"k_": ['iTot'],
	"k" : ['iM', 'iB','iT'],
	
  "n_":['tot'],
	"n":['byg','Dag','Dep','Elek','Farlig','Glas','Have','Jord','Metal','Org','Papir','PVC','Plast','Rest','Slam','Trae','Tekst'],
	
  "pens_":['Pension'],
	"Pens":['PensX','Kap','Alder'],

	"portfolio": ['NetFinAssets','Debt','Mortgages','Gold','Deposits','Pension','ForeignEquity','Equity','Bonds'],
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

