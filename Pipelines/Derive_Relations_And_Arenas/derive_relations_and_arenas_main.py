"""Main script for the Derive_Relations_And_Arenas pipeline.

To run this script, use the following command from the project root:
    uv run python -m Pipelines.Derive_Relations_And_Arenas.derive_relations_and_arenas_main
"""

## IMPORTS ##
import polars as pl
import pandas as pd
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

OUTPUT_DIR_COUNTS = f"{OUTPUT_DIR_WORDS_BASE}/counts.xlsx"

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
print(stop_words)
stop_words.update(stopwords_extension)
stop_words.update(maybe_stopwords)

min_word_count = 1000
min_text_count = 900
## _______________________ ##


## HELPER FUNCTIONS ##
def _count_used_words(
    input_df: pl.DataFrame,
    min_word_count=min_word_count,
    min_text_count=min_text_count
    # logger: logging.Logger
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    word_counts = Counter()
    text_counts = Counter()

    texts = input_df["msg_tekst"]
    word_counts = Counter()
    text_counts = Counter()

    for text in texts:
        all_words = re.findall(r"\b[\w']+\b", text.lower())
        new_words = [word for word in all_words if word not in stop_words]
        new_texts = {word for word in all_words if word not in stop_words}
        word_counts.update(new_words)
        text_counts.update(new_texts)

    filtered_words = Counter({
        word: count
        for word, count in word_counts.items()
        if count > min_word_count
    })

    filtered_texts = Counter({
        word: count
        for word, count in text_counts.items()
        if count > min_text_count
    })

    return filtered_words.most_common(), filtered_texts.most_common()

def _combine_to_xlsx(
    counts: dict[str, tuple[list[tuple[str, int]], list[tuple[str, int]]]],
    output_path: str | Path,
) -> None:
    columns = {}
    for name, (words, conversations) in counts.items():
        columns[(name, "Words")] = pd.Series(
            [word for word, _ in words], dtype="string"
        )
        columns[(name, "Conversations")] = pd.Series(
            [word for word, _ in conversations], dtype="string"
        )

    table = pd.DataFrame(columns)
    table.columns = range(table.shape[1])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        table.to_excel(
            writer,
            sheet_name="Counts",
            startrow=2,
            header=False,
            index=False,
        )

        workbook = writer.book
        sheet = writer.sheets["Counts"]

        group_header = workbook.add_format({
            "bold": True, "align": "center", "right": 5
        })
        subheader = workbook.add_format({"bold": True})
        subheader_right = workbook.add_format({"bold": True, "right": 5})
        right_border = workbook.add_format({"right": 5})

        for group_index, name in enumerate(counts):
            left_col = group_index * 2
            right_col = left_col + 1

            sheet.merge_range(0, left_col, 0, right_col, name, group_header)
            sheet.write(1, left_col, "Words", subheader)
            sheet.write(1, right_col, "Conversations", subheader_right)

            sheet.set_column(left_col, right_col, 24)

            # Format every cell on the group's right edge, including blanks.
            for row_index in range(len(table)):
                value = table.iloc[row_index, right_col]
                excel_row = row_index + 2
                if pd.isna(value):
                    sheet.write_blank(excel_row, right_col, None, right_border)
                else:
                    sheet.write_string(
                        excel_row, right_col, value, right_border
                    )

        sheet.freeze_panes(2, 0)
## ________________ ##


## MAIN FUNCTION ##
def main() -> None:
    """Run the full Derive_Relations_And_Arenas pipeline."""
    df = pl.read_parquet(INPUT_DIR_ALL_CONVOS)
    groups = {
        "All": df,
        "Boys": df.filter(pl.col("gender") == "male"),
        "Girls": df.filter(pl.col("gender") == "female"),
        "Pre-teens": df.filter(pl.col("age_group") == "pre teen"),
        "Early-teens": df.filter(pl.col("age_group") == "early teen"),
        "Teens": df.filter(pl.col("age_group") == "teen"),
    }

    counts = {
        name: _count_used_words(group_df)
        for name, group_df in groups.items()
    }
    _combine_to_xlsx(counts, OUTPUT_DIR_COUNTS)

## CALL OF MAIN FUNCTION ##
if __name__ == "__main__":
    main()
