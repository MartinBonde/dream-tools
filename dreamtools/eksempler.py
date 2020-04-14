import dreamtools as dt


# Åben gdx fil
gdx = dt.Gdx("test.gdx")
gdx.export("export.gdx")

# Hent sets fra gdx
s, i, t = gdx.get("s", "i", "t")

# Hent variable fra gdx
qY, qBNP, qI_s = gdx.get("qY", "qBNP", "qI_s")

# Plot variabel
qBNP.plot()

# Plot variabel i begrænset periode
qBNP.plot(2010, 2020)

# Plot variabel begrænset til et enkelt element eller set
qY["tot"].plot()
qY[s].plot()

# Plot flere variable sammen
dt.plot(qY[s], qBNP, start=2010, end=2020)

# Hvis en serie har flere dimensioner, skal vi skrive .loc for at begrænse med på mere end første dimension. Fx:
qI_s.loc[i,s]

# Vi kan stadig begænse på første dimension uden .loc
qI_s[i]

# Hvis vi begrænser første dimnension til et enkelt element kan vi derefter begrænse anden dimension uden .loc
qI_s["IB"]["bol"] == qI_s.loc["IB","bol"]
# Generelt er det en god idé at skrive .loc hvis man er i tvivl
# (det gør det eksplicit at vi vil bruge 'label based indexing' fremfor et numerisk indeks)

# Sæt globalt standard startår og slutår
dt.time(2010, 2040)

# Gang to variable sammen
vY = gdx["pY"]*qY
# Divider med variablens værdi i 2010
vY /= vY[:,'2010']
# Plot med en overskrift tilføjet
vY[s].plot(title="pY*qY, index=2010")

# Check at der eksisterer en 'images' mappe og opret den hvis den ikke gør
import os
if not os.path.exists("images"):
  os.mkdir("images")

# Plot som statisk billed, fx som svg, pdf eller png. Bemærk at mange billed-formater kan åbnes direkte i PyCharm
qBNP.plot(file="images/plot.svg")
# Dette kræver at man har Orca installeret - spørg evt. Lucas eller undertegnede for hjælp


# Skift om der renderes til browser, pdf, png etc. som standard
import plotly.io as pio
pio.renderers.default = "browser"


# Lav ny GamsPandasDatabase og alias af metoder til at lave nye symboler
db = dt.GamsPandasDatabase()
Par, Var, Set = db.create_parameter, db.create_variable, db.create_set

# Definer sets
t = Set("t", range(2000, 2020), "Årstal")
s = Set("s", ["tjenester", "fremstilling"], "Brancher")
st = Set("st", [s, t], "Branche x år dummy")
sub = Set("sub", ["tjenester"], "Subset af brancher", domains=["s"])

one2one = Set("one2one", [(2010, 2015), (2011, 2016)], "1 til 1 mapping", domains=["t", "t"])

one2many = Set("one2many",
               [("tot", "tjenester"), ("tot", "fremstilling")],
               "1 til mange mapping", domains=["*", "s"],
               )

# Definer nye parametre og variable, gerne ud fra sets
gq = Par("gq", None, "Produktivitets-vækst", 0.01)
fq = Par("fp", t, "Vækstkorrektionsfaktor", (1 + 0.01) ** (t - 2010))
d = Par("d", st, "Dummy")
y = Var("y", [s, t], "Produktion")
p = Var("p", [s, t], "Pris")

# Assignment
y["tjenester"], y["fremstilling"] = 7, 3
p = 100  # Hov, vi kom til at overskrive vores p parameter!
p = db["p"]  # Den kan heldigivs findes igen
p[:] = 100

# Beregner med symboler som deler sets sker uden problemer
y *= fq
print(p * y)