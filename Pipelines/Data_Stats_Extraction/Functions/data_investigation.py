"""Investigate descriptive statistics for cleaned ACT data."""

## IMPORTS ##
# Standard library
import logging

# External
import polars as pl 

## _______ ##


## HELPER FUNCTIONS ##

## ________________ ##

## MAIN FUNCTIONALITY ##
def data_investigation_messages(
    input_dir: str, 
    output_dir: str,
    logger: logging.Logger,
):
    stats_dict = {}
    pdf = pl.read_parquet(input_dir)

    # Include dataset information
    stats_dict["Dataset_type"] = "Original messages"
    logger.info("Dataset type has been added to stats.")
    logger.info(f"Current stats_dict: {stats_dict}")
    logger.info("="*20)

    # Include message and conversation stats
    number_of_messages = pdf.height
    number_of_incoming = pdf.filter(
        pl.col("is_incoming") == True
    ).height
    number_of_outgoing = pdf.filter(
        pl.col("is_incoming") == False
    ).height

    logger.info(
        f"Message stats:\
            \nTotal: {number_of_messages}\
            \nIncoming: {number_of_incoming}\
            \nOutgoing: {number_of_outgoing}\
            \nDifference in total and in + out: {
                number_of_messages 
                - number_of_incoming
                - number_of_outgoing
            }"
        )
    logger.info("="*20)

    message_stats = {
        "Total": number_of_messages,
        "Incoming": number_of_incoming,
        "Outgoing": number_of_outgoing
    }

    conversations = pdf.group_by("conversation_code").agg(
        pl.len().alias("total_messages"),
        pl.col("is_incoming").sum().alias("incoming_messages"),
        (~pl.col("is_incoming")).sum().alias("outgoing_messages")
    )

    number_of_conversations = conversations.height
    avg_number_of_messages = round(conversations["total_messages"].mean(), 2)
    med_number_of_messages = conversations["total_messages"].median()
    avg_number_of_incoming = round(conversations["incoming_messages"].mean(), 2)
    med_number_of_incoming = conversations["incoming_messages"].median()
    avg_number_of_outgoing = round(conversations["outgoing_messages"].mean(), 2)
    med_number_of_outgoing = conversations["outgoing_messages"].median()

    logger.info(
        f"Conversation stats:\
        \nTotal: {number_of_conversations}\
        \nAvg messages: {avg_number_of_messages}\
        \nMed messages: {med_number_of_messages}\
        \nAvg incoming: {avg_number_of_incoming}\
        \nMed incoming: {med_number_of_incoming}\
        \nAvg outgoing: {avg_number_of_outgoing}\
        \nMed outgoing: {med_number_of_outgoing}"
    )
    logger.info("="*20)

    conversation_stats = {
        "Total": number_of_conversations,
        "Messages": {
            "Average": avg_number_of_messages,
            "Median": med_number_of_messages
        },
        "Incoming": {
            "Average": avg_number_of_incoming,
            "Median": med_number_of_incoming
        },
        "Outgoing": {
            "Average": avg_number_of_outgoing,
            "Median": med_number_of_outgoing
        }
    }

    stats_dict["stats"] = {
        "Message_stats": message_stats,
        "Conversation_stats": conversation_stats
    }
    logger.info("Stats for messages and conversations have been added to stats_dict")
    logger.info(f"Current stats_dict: {stats_dict}")
    logger.info("="*20)

    # Include distributions
    number_of_messages_preteen = pdf.filter(
        pl.col("age_group") == "pre teen"
    ).height 
    number_of_messages_earlyteen = pdf.filter(
        pl.col("age_group") == "early teen"
    ).height 
    number_of_messages_teen = pdf.filter(
        pl.col("age_group") == "teen"
    ).height 

    age_distribution = {
        "Pre teen (10-12)": number_of_messages_preteen,
        "Early teen (13-14)": number_of_messages_earlyteen,
        "Teen (15-17)": number_of_messages_teen
    }

    logger.info(
        f"Age distribution:\
        \nPre teen (10-12): {number_of_messages_preteen}\
        \nEarly teen (13-14): {number_of_messages_earlyteen}\
        \nTeen (15-17): {number_of_messages_teen}\
        \nDifference in total messages and pre teens, early teens and teens: {
            number_of_messages 
            - number_of_messages_preteen 
            - number_of_messages_earlyteen 
            - number_of_messages_teen
        }"
    )
    logger.info("="*20)

    number_of_messages_male = pdf.filter(
        pl.col("gender") == "male"
    ).height
    number_of_messages_female = pdf.filter(
        pl.col("gender") == "female"
    ).height
    number_of_messages_other = pdf.filter(
        pl.col("gender") == "other"
    ).height

    gender_distribution = {
        "Male": number_of_messages_male,
        "Female": number_of_messages_female,
        "Other": number_of_messages_other
    }

    number_of_messages_no_gender = pdf.filter(
        pl.col("gender").is_null()
    ).height

    logger.info(
        f"Gender distribution:\
        \nMale: {number_of_messages_male}\
        \nFemale: {number_of_messages_female}\
        \nOther: {number_of_messages_other}\
        \nNo gender: {number_of_messages_no_gender}\
        \nDifference in total messages and males, females and other: {
            number_of_messages 
            - number_of_messages_male 
            - number_of_messages_female 
            - number_of_messages_other
        }"
    )
    logger.info("="*20)

    stats_dict["Distributions"] = {
        "Age": age_distribution,
        "Gender": gender_distribution
    }

    logger.info("Distributions for messages have been added to stats_dict")
    logger.info(f"Current stats_dict: {stats_dict}")
    logger.info("="*20)


def data_investigation_collapsed_messages(
    input_dir: str, 
    output_dir: str,
    logger: logging.Logger,
):
    pdf = pl.read_parquet(input_dir)
    print(pdf.head(5))

def data_investigation_conversations(
    input_dir: str, 
    output_dir: str,
    logger: logging.Logger,
):
    pdf = pl.read_parquet(input_dir)
    print(pdf.head(5))

## __________________ ##