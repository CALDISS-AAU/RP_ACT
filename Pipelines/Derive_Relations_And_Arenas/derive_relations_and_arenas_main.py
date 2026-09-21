"""Main script for the Derive_Relations_And_Arenas pipeline.

To run this script, use the following command from the project root:
    uv run python -m Pipelines.Derive_Relations_And_Arenas.derive_relations_and_arenas_main
"""

## IMPORTS ##
import polars as pl
from collections import Counter
import re 
import csv
from pathlib import Path
from spacy.lang.da import Danish
# Internal
from Shared_Functions.logger_functionality import *
## _______ ##


## STATIC VARIABLES ##
# Directories - input
INPUT_DIR_ALL_CONVOS = "./Data/Clean_Combine/conversation_data_without_advisors.parquet"

# Directories - internal output
# OUTPUT_DIR_AAA = "Pipelines/Derive_Relations_And_Arenas/Data/xxx.zzz"

# Directories - global output
OUTPUT_DIR_WORDS_BASE = "./Data/Derive_Relations_And_Arenas"
OUTPUT_DIR_UNIQUE_WORDS_ALL = f"{OUTPUT_DIR_WORDS_BASE}/word_counts_all.csv"
OUTPUT_DIR_UNIQUE_WORDS_BOYS = f"{OUTPUT_DIR_WORDS_BASE}/word_counts_boys.csv"
OUTPUT_DIR_UNIQUE_WORDS_GIRLS = f"{OUTPUT_DIR_WORDS_BASE}/word_counts_girls.csv"
OUTPUT_DIR_UNIQUE_WORDS_PRETEEN = f"{OUTPUT_DIR_WORDS_BASE}/word_counts_preteen.csv"
OUTPUT_DIR_UNIQUE_WORDS_EARLYTEEN = f"{OUTPUT_DIR_WORDS_BASE}/word_counts_earlyteen.csv"
OUTPUT_DIR_UNIQUE_WORDS_TEEN = f"{OUTPUT_DIR_WORDS_BASE}/word_counts_teen.csv"

# Other
maybe_stopwords = (
    'me', # Engelsk 'mig selv'?
    'migselv', # Er jeg selv en aktør?
    'hinanden',
    # 'folk',
    # 'piger',
    # 'pigerne,
    # 'drenge',
    # 'drengene',
    'kærester',
    'eks', # Måske tidligere kæreste, måske eksempel
    'børn',
    'barn',
    'mand', # Snakker de med/om en mand, skal de være en mand, eller er det en sætningsforstærker?
    'mennesker',
    'mors', # Måske stavefejl af mor?
    'børnetelefonen', # Jeg ved ikke nok om materialet til at vide om dette er en aktør
    'rådgiver', # Snakker de med en rådgiver et sted, rådgiver folk dem, eller er det meta-snak i forbindelse med børnetelefonens rådgivere?
    # 'person',
    'lære', # Er dette bare en stavefejl for lærer? For i så fald er det vel en aktør?
    'politiet', # Snakker de med politiet, vil de ikke i kontakt med politiet, eller ...?
    'telefon', # Sker det på telefonen? For så er det vel en lokation?
    'besked', # Samme som telefon
    'beskeder',
    'snap', # Snakker de om snap, eller sker der ting på snap?
    'flytte', # Mellemstadie mellem to lokationer?
    'loc', # Anonymiseret lokation?
    'location', # Samme som loc
    'bar', # Har de været i byen, har de ikke noget tøj på eller er det en stavefejl af bare?
    'tanken', # Er de på tanken (gas station) eller har de tanken (idea)?
    'fest', # Sker der noget til en fest?
    'fysisk', # vs online, eller vs psykisk?
    # 'online', # vs fysisk?
    'pause', # Frikvarteret? For så er det vel indirekte på skolen et sted?
    'ferie', # Imens de var på ferie? 
)
stopwords_extension = (
    'bare','ja','godt','ik','tak',
    'tror','når','nej','hej','virkelig',
    'siger','rigtig','år','gerne','snakke',
    'går','helt','okay','hele','ting',
    'sagde','altså','gange','pige','dag',
    'tid','gang','haft','gå','hjælp',
    'får','tænker','se','tiden','skrive',
    'lyst','sagt','brug','lide','tage',
    'sur','ked','ret','måde','komme',
    'hjælpe','2','snakker','skriver','prøve',
    'fik','to','faktisk','rigtigt','blevet',
    'prøver','fået','dårligt','væk','prøvet',
    'forstår','snakket','ser','sidste','finde',
    '3','bedre','tager','gik','vide',
    'lang','dreng','ok','hos','bedste',
    'bor','mega','tale','fortælle','hjælpen',
    'tusind','giver','tænke','skrev','gider',
    'snart','fint','glad','dage','ikk',
    'slet','1','name','morgen','lyder',
    'sove','sker','ondt','tør','hjælper',
    'skrevet','holde','15','første','13',
    '14','angst','tog','mening','høre',
    'måneder','kender','problemer','begyndte','gode',
    'begyndt','stoppe','træt','ide','gået',
    'dårlig','hader','12','taget','håber',
    'elsker','5','spise','mener','forhold',
    'per','hvilket','par','4','normalt',
    'aften','mærke','uge','laver','osv',
    'engang','stop','fx','føles','forkert',
    'give','grund','inde','hedder','set',
    'spørge','spørger','vis','sød','værre',
    'spurgte','muligt','hårdt','følelser','inden',
    'tænkt','ude','sidder','16','virker',
    '10','krop','lade','rart','begynder',
    'ringe','seng','hved','sikker','sket',
    'snakkede','videre','fortalt','dagen','forstå',
    'snak','problem','kommet','7','klar',
    'del','ogs','mod','forklare','uger',
    'svare','står','skete','pga','plejer',
    'sted','and','starte','selvmordstanker','overhovedet',
    'føler','bange','svært','selvskade','tanker',
    'liv','undskyld','sex','helst','bruge',
    'egentlig','sure','d','huske','både',
    'sin','fortæller','17','galt','nummer',
    'vildt','spiser','finder','startede','fylder',
    '11','særlig','leve','spørgsmål','tæt',
    'råd','rundt','6','nye','mad',
    'dø','timer','gammel','skyld','holder',
    'vidste','9','tidspunkt','græde','ca',
    'stor','ske','ligger','nervøs','hurtigt',
    'længe','selvfølgelig','føle','fandt','hoved',
    'tænkte','ha','betyder','a','vel',
    'langt','svar','slemt','følte','fortalte',
    'bo','kontakt','prøvede','side','8',
    'selvmord','taler','adhd','sende','gamle',
    'seriøst','stille','problemet','skændes','måned',
    'masse','bekymret','sårn','samtale','slå',
    'syg','sover','sikkert','græder','ligeglad',
    'råber','ej','idag','klare','hey',
    'lytte','gjordt','ender','samtalen','troede',
    'talt','hørt','bruger','sidst','gad',
    'især','mmærkeligt','18','ofte','penge',
    'vente','lytter','time','ønske','depression',
    'hjalp','nå','gravid','hovedet','livet',
    'hvært','årlig','nød','tro','fucking',
    'udenfor','skilt','spurgt','savner','situation',
    'måtte','kigger','nat','dumt','fald',
    'fundet','kl','hallo','super','ord',
    'hjulpet','tankerne','ringer','slået','skade',
    'søvn','maven','ønsker','natten','følelse',
    'kæmpe','the','ro','meste','slog','små',
    'slår','generelt','usikker','tryg','my','ka',
    'dele','rigtige','tøj','indlagt','mobbet',
    'hellere','minutter','vej','bedst','starten',
    'sendt','billeder','pinligt','sygt','stopper',
    'tur','gav','tingene','dum','stort',
    'tvivl','alder','stresset','godnat','ældre',
    'stedet','gruppe','oppe','stå','ugen',
    'passer','møde','lov','ærlig','100',
    'uvenner','højt','stoler','anderledes','ku',
    'skifte','tyk','pludselig','forskellige','sjovt',
    'cirka','tag','nærmest','oplevet','spiseforstyrrelse',
    'konstant','bad','desværre','sad','altså',
    'mentalt','ubehagelige','piller','mulighed','køre',
    'dårlige','presset','søde','store','handler',
    'sjov','endte','akavet','givet','sidde',
    'præcis','overvejer','forelsket','frem','nemt',
    'it','ærligt','steder','overfor','passe',
    'ensom','læse','fat','ændre','kort',
    'mensturation','arbejder','sender','20','spille',
    'stoppede','startet','navn','irriterende','nødt',
    'sætte','sommerferien','miste','åbne','verden',
    'fejl','hører','vise','stress','lort',
    'spiller','holdt','tætte','medicin','skidt',
    'tre','smule','kalder','kigge','sku',
    '30','jaer','syndes','mente','musik',
    'grunden','imorgen','stoppet','blir','fuld',
    'nemmere','ligge','snakken','mistet','træls',
    'f','fuck','hjerte','eget','kontakte',
    'mandag','svarer','hva','følt','igår',
    'starter','psykisk','endelig','autisme','sent',
    'idk','fedt','hår','normal','jammen',
    'trist','slags','sov','stole','stod',
    'underligt','lavede','dør','mærkelig','in',
    'institution','farvel','overvejet','ende','dejligt',
    'klokken','alting','halvt','sendte','weekend',
    'overskue','svære','dræbe','jamen','alså',
    'menstruation','mærkeligt','hvert','årlig','ubehageligt',
    
)

nlp = Danish()
stop_words = set(nlp.Defaults.stop_words)
stop_words.update(stopwords_extension)
stop_words.update(maybe_stopwords)
## _______________________ ##


## HELPER FUNCTIONS ##
def _count_used_words(
    input_df: pl.DataFrame,
    output_dir: str,
    min_count=1,
    # logger: logging.Logger
) -> None:
    texts = input_df["msg_tekst"]
    word_counts = Counter()

    for text in texts:
        new_words = re.findall(r"\b[\w']+\b", text.lower())
        new_words = [word for word in new_words if word not in stop_words]
        word_counts.update(new_words)

    filtered_words = Counter({
        word: count
        for word, count in word_counts.items()
        if count > min_count
    })

    output_path = Path(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["word", "count"])
        writer.writerows(filtered_words.most_common())
## ________________ ##


## MAIN FUNCTION ##
def main() -> None:
    """Run the full Derive_Relations_And_Arenas pipeline."""
    # All
    df = pl.read_parquet(INPUT_DIR_ALL_CONVOS)
    _count_used_words(
        input_df = df,
        output_dir = OUTPUT_DIR_UNIQUE_WORDS_ALL,
        min_count=1000
    )
    
    # Boys
    df_boys = df.filter(
        pl.col("gender").eq('male')
    )
    _count_used_words(
        input_df = df_boys,
        output_dir = OUTPUT_DIR_UNIQUE_WORDS_BOYS,
        min_count=1000
    )

    # Girls
    df_girls = df.filter(
        pl.col("gender").eq('female')
    )
    _count_used_words(
        input_df = df_girls,
        output_dir = OUTPUT_DIR_UNIQUE_WORDS_GIRLS,
        min_count=1000
    )

    # Pre-teens
    df_preteen = df.filter(
        pl.col("age_group").eq('pre teen')
    )
    _count_used_words(
        input_df = df_preteen,
        output_dir = OUTPUT_DIR_UNIQUE_WORDS_PRETEEN,
        min_count=1000
    )

    # Early-teens
    df_earlyteen = df.filter(
        pl.col("age_group").eq('early teen')
    )
    _count_used_words(
        input_df = df_earlyteen,
        output_dir = OUTPUT_DIR_UNIQUE_WORDS_EARLYTEEN,
        min_count=1000
    )

    # Teens
    df_teen = df.filter(
        pl.col("age_group").eq('teen')
    )
    _count_used_words(
        input_df = df_teen,
        output_dir = OUTPUT_DIR_UNIQUE_WORDS_TEEN,
        min_count=1000
    )

## CALL OF MAIN FUNCTION ##
if __name__ == "__main__":
    main()
