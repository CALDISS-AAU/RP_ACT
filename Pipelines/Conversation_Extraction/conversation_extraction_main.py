"""Main script for the Conversation_Extraction pipeline."""

"""Run the ACT clean-and-combine pipeline. 
   The script inspects raw conversation data, 
       cleans and combines the raw CSV files, 
       adds derived variables, exports the final dataset, 
       and rebuilds the full pipeline log. 

   Run from the ACT folder with: 
       python -m Pipelines.Conversation_Extraction.conversation_extraction_main 
"""
## IMPORTS ##
# Internal
from Shared_Functions.logger_functionality import *
from .Functions.data_investigation import investigate_data_stats
from .Functions.get_random_conversations import export_random_conversation_examples

## _______ ##

## STATIC VARIABLES ##
# Directories - input
INPUT_DIR_CONVERSATIONS = "/work/ACT/Data/clean_combine/conversation_data.parquet" 

# Directories - internal output
OUTPUT_DIR_DATA_INVESTIGATION_PNGS = "Pipelines/Conversation_Extraction/Data/pngs" 

# Directories - global output
OUTPUT_DIR_FINAL_DATA_INVESTIGATION = "/work/RP_ACT/Output/Conversation_Extraction/data_investiogation_stats.docx"
OUTPUT_DIR_CONVERSATION_EXAMPLES = "/work/RP_ACT/Output/Conversation_Extraction/Conversation_Examples"

# Directories - logs
OUTPUT_DIR_LOG_FULL_PIPELINE = "./Pipelines/Conversation_Extraction/Logs/full_conversation_extraction_pipeline.log"
OUTPUT_DIR_LOG_DATA_INVESTIGATION = "./Pipelines/Conversation_Extraction/Logs/data_investigation.log"
OUTPUT_DIR_LOG_CONVERSATION_EXAMPLES = "./Pipelines/Conversation_Extraction/Logs/conversation_examples.log"

# Other
TOPIC_CONTACT_REASONS_FROM_REASEARCHERS = {
    "familie_og_omsorgsrelationer": [
        "alkohol- eller stofproblemer i familien",
        "anbringelse",
        "forældre skændes",
        "forældres krav og forventninger",
        "fysisk sygdom i familien",
        "psykisk sygdom i familien",
        "hverdagsrutiner og opdragelse",
        "kontakt med familieretshuset",
        "kontakt med kommunen",
        "løbet hjemmefra/smidt ud hjemmefra",
        "sammenbragte og nye familier",
        "samvær/bopæl og forældremyndighed",
        "udfordringer ved skilsmisse",
        "utryg/konfliktfyldt relation til forældre",
        "forsømmelse/vanrøgt",
        "fattigdom/knappe økonomiske ressourcer",
        "dødsfald/sorg",
        "religion/kultur",
    ],
    "venskaber_fællesskaber_og_sociale_relationer": [
        "ensomhed",
        "konflikter/bekymringer i venskaber",
        "mobning i det fysiske rum",
        "digital mobning",
        "trivsel i klassen/skolen",
        "relation til lærer/pædagog",
        "gaming",
        "brug af sociale medier",
    ],
    "skole_og_uddannelse": [
        "faglige problemer/lektier",
        "skolefravær",
        "skoleskift",
        "stress/præstationsangst",
        "arbejdsrelaterede udfordringer",
    ],
    "psykisk_trivsel_mental_sundhed_og_social_adfærd": [
        "angst",
        "depression",
        "selvmordstanker",
        "selvmordsplaner/selvmordsforsøg",
        "selvskadende adfærd",
        "spiseforstyrrelser",
        "anden psykisk lidelse/diagnose",
        "ADHD/ADD",
        "autisme",
        "udadreagerende adfærd",
        "kontakt med psykiatrien",
    ],
    "krop_identitet_seksualitet_og_sundhed": [
        "kroppens udseende og udvikling",
        "kønsidentitet",
        "seksuel orientering",
        "graviditet",
        "kønssygdomme",
        "sex og seksuel praksis",
        "flirt/forelskelse",
        "problemer med kæreste/eks",
        "fysisk sygdom hos barnet",
        "brug af alkohol",
        "brug af nikotin",
        "brug af stoffer",
    ],
    "krænkelser_og_vold": [
        "afpresning/trusler på nettet",
        "billed-/filmdeling",
        "digital seksuel krænkelse",
        "fysisk seksuel krænkelse",
        "fysisk vold",
        "psykisk vold",
        "negativ social kontrol",
        "krig",
    ],
}

# TOPIC_CONTACT_REASONS = {
#     "familie_og_omsorgsrelationer": [
#         "alkohol- eller stofproblemer i familien",
#         "anbringelse",
#         "forældre skændes",
#         "forældres krav og forventninger",
#         "fysisk sygdom i familien - anden",
#         "fysisk sygdom i familien - det kom ikke frem",
#         "fysisk sygdom i familien - primære omsorgspersoner",
#         "fysisk sygdom i familien - øvrig familie",
#         "psykisk sygdom i familien - anden",
#         "psykisk sygdom i familien - det kom ikke frem",
#         "psykisk sygdom i familien - primære omsorgspersoner",
#         "psykisk sygdom i familien - øvrig familie",
#         "hverdagsrutiner og opdragelse",
#         "kontakt med familieretshuset",
#         "kontakt med kommunen",
#         "løbet hjemmefra/smidt ud hjemmefra",
#         "sammenbragte og nye familier",
#         "samvær, bopæl og forældremyndighed",
#         "udfordringer ved skilsmisse",
#         "utryg/konfliktfyldt relation til forælder",
#         "forsømmelse/vanrøgt - anden",
#         "forsømmelse/vanrøgt - primære omsorgspersoner",
#         "forsømmelse/vanrøgt - øvrig familie",
#         "fattigdom/knappe økonomiske ressourcer",
#         "dødsfald og/eller sorg",
#         "religion/kultur",
#     ],
#     "venskaber_fællesskaber_og_sociale_relationer": [
#         "ensomhed",
#         "konflikter/bekymringer i venskaber",
#         "mobning i det fysiske rum",
#         "digital mobning",
#         "trivsel i klassen/skolen",
#         "relation til lærer/pædagog mv.",
#         "gaming",
#         "brug af sociale medier",
#     ],
#     "skole_og_uddannelse": [
#         "faglige problemer/lektier",
#         "skolefravær",
#         "skoleskift",
#         "stress/præstationsangst",
#         "arbejdsrelaterede udfordringer",
#     ],
#     "psykisk_trivsel_mental_sundhed_og_social_adfærd": [
#         "angst",
#         "bange/nervøs",
#         "depression",
#         "selvmordstanker",
#         "selvmordsplaner/selvmordsforsøg",
#         "selvskadende adfærd",
#         "spiseforstyrrelser",
#         "anden psykisk lidelse / diagnose",
#         "adhd / add",
#         "autisme",
#         "udadreagerende adfærd",
#         "kontakt med psykiatrien",
#     ],
#     "krop_identitet_seksualitet_og_sundhed": [
#         "kroppens udseende og udvikling",
#         "kønsidentitet",
#         "seksuel orientering",
#         "graviditet",
#         "kønssygdomme",
#         "sex og seksuel praksis",
#         "flirt/forelskelse",
#         "problemer med kæreste/eks",
#         "fysisk sygdom hos barnet",
#         "brug af alkohol",
#         "brug af nikotin",
#         "brug af stoffer",
#     ],
#     "krænkelser_og_vold": [
#         "afpresning/trusler på nettet",
#         "billed-/filmdeling",
#         "digital seksuel krænkelse",
#         "fysisk seksuel krænkelse - anden",
#         "fysisk seksuel krænkelse - det er barnet selv, der udsætter andre",
#         "fysisk seksuel krænkelse - det kom ikke frem",
#         "fysisk seksuel krænkelse - jævnaldrende",
#         "fysisk seksuel krænkelse - kæreste/ekskæreste",
#         "fysisk seksuel krænkelse - primære omsorgspersoner",
#         "fysisk seksuel krænkelse - øvrig familie",
#         "fysisk vold - anden",
#         "fysisk vold - det er barnet selv, der udsætter andre",
#         "fysisk vold - det kom ikke frem",
#         "fysisk vold - jævnaldrende",
#         "fysisk vold - kæreste/ekskæreste",
#         "fysisk vold - primære omsorgspersoner",
#         "fysisk vold - øvrig familie",
#         "psykisk vold - anden",
#         "psykisk vold - det er barnet selv, der udsætter andre",
#         "psykisk vold - det kom ikke frem",
#         "psykisk vold - jævnaldrende",
#         "psykisk vold - kæreste/ekskæreste",
#         "psykisk vold - primære omsorgspersoner",
#         "psykisk vold - øvrig familie",
#         "negativ social kontrol",
#         "krig",
#     ],
#     "udenfor_kategori": [
#         "udenfor kategori",
#     ],
# }

TOPIC_CONTACT_REASONS = {
    "familie": [
        "alkohol- eller stofproblemer i familien",
        "anbringelse",
        "forældre skændes",
        "forældres krav og forventninger",
        "fysisk sygdom i familien - anden",
        "fysisk sygdom i familien - det kom ikke frem",
        "fysisk sygdom i familien - primære omsorgspersoner",
        "fysisk sygdom i familien - øvrig familie",
        "psykisk sygdom i familien - anden",
        "psykisk sygdom i familien - det kom ikke frem",
        "psykisk sygdom i familien - primære omsorgspersoner",
        "psykisk sygdom i familien - øvrig familie",
        "hverdagsrutiner og opdragelse",
        "kontakt med familieretshuset",
        "kontakt med kommunen",
        "løbet hjemmefra/smidt ud hjemmefra",
        "sammenbragte og nye familier",
        "samvær, bopæl og forældremyndighed",
        "udfordringer ved skilsmisse",
        "utryg/konfliktfyldt relation til forælder",
        "forsømmelse/vanrøgt - anden",
        "forsømmelse/vanrøgt - primære omsorgspersoner",
        "forsømmelse/vanrøgt - øvrig familie",
        "fattigdom/knappe økonomiske ressourcer",
        "dødsfald og/eller sorg",
        "religion/kultur",
    ],
    "venskaber": [
        "ensomhed",
        "konflikter/bekymringer i venskaber",
        "mobning i det fysiske rum",
        "digital mobning",
        "trivsel i klassen/skolen",
        "relation til lærer/pædagog mv.",
        "gaming",
        "brug af sociale medier",
    ],
    "skole": [
        "faglige problemer/lektier",
        "skolefravær",
        "skoleskift",
        "stress/præstationsangst",
        "arbejdsrelaterede udfordringer",
    ],
    "psykisk_trivsel": [
        "angst",
        "bange/nervøs",
        "depression",
        "selvmordstanker",
        "selvmordsplaner/selvmordsforsøg",
        "selvskadende adfærd",
        "spiseforstyrrelser",
        "anden psykisk lidelse / diagnose",
        "adhd / add",
        "autisme",
        "udadreagerende adfærd",
        "kontakt med psykiatrien",
    ],
    "krop": [
        "kroppens udseende og udvikling",
        "kønsidentitet",
        "seksuel orientering",
        "graviditet",
        "kønssygdomme",
        "sex og seksuel praksis",
        "flirt/forelskelse",
        "problemer med kæreste/eks",
        "fysisk sygdom hos barnet",
        "brug af alkohol",
        "brug af nikotin",
        "brug af stoffer",
    ],
    "krænkelser": [
        "afpresning/trusler på nettet",
        "billed-/filmdeling",
        "digital seksuel krænkelse",
        "fysisk seksuel krænkelse - anden",
        "fysisk seksuel krænkelse - det er barnet selv, der udsætter andre",
        "fysisk seksuel krænkelse - det kom ikke frem",
        "fysisk seksuel krænkelse - jævnaldrende",
        "fysisk seksuel krænkelse - kæreste/ekskæreste",
        "fysisk seksuel krænkelse - primære omsorgspersoner",
        "fysisk seksuel krænkelse - øvrig familie",
        "fysisk vold - anden",
        "fysisk vold - det er barnet selv, der udsætter andre",
        "fysisk vold - det kom ikke frem",
        "fysisk vold - jævnaldrende",
        "fysisk vold - kæreste/ekskæreste",
        "fysisk vold - primære omsorgspersoner",
        "fysisk vold - øvrig familie",
        "psykisk vold - anden",
        "psykisk vold - det er barnet selv, der udsætter andre",
        "psykisk vold - det kom ikke frem",
        "psykisk vold - jævnaldrende",
        "psykisk vold - kæreste/ekskæreste",
        "psykisk vold - primære omsorgspersoner",
        "psykisk vold - øvrig familie",
        "negativ social kontrol",
        "krig",
    ],
    "andet": [
        "udenfor kategori",
    ],
}
## _______________________ ##

## HELPER FUNCTIONS ##
## ________________ ##

## MAIN FUNCTION ##
def main() -> None: 
    """Run all conversation-extraction pipeline steps."""
    investigate_data_stats(
        INPUT_DIR_CONVERSATIONS,
        OUTPUT_DIR_FINAL_DATA_INVESTIGATION,
        output_dir_pngs=OUTPUT_DIR_DATA_INVESTIGATION_PNGS,
        topic_contact_reasons=TOPIC_CONTACT_REASONS,
        logger=setup_logger(
            output_dir_log=OUTPUT_DIR_LOG_DATA_INVESTIGATION,
            logger_name="conversation_extraction.data_inspection",
        ),
        bottom_percent=1/6*100,
        wordcount_interval_size=200,
        wordcount_stop_at=2000,
    )

    export_random_conversation_examples(
        INPUT_DIR_CONVERSATIONS,
        OUTPUT_DIR_CONVERSATION_EXAMPLES,
        5,
        logger=setup_logger(
            output_dir_log=OUTPUT_DIR_LOG_CONVERSATION_EXAMPLES,
            logger_name="conversation_extraction.conversation_examples",
        ),
    )


## _____________ ##

## CALL OF MAIN FUNCTION ##
if __name__ == "__main__":
    main()