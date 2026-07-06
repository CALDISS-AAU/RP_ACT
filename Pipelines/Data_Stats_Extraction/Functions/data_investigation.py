"""Investigate descriptive statistics for cleaned ACT data."""

## IMPORTS ##
# Standard library
import logging

# External
import polars as pl 

## _______ ##


## HELPER FUNCTIONS ##
def _avg_med_min_max(df:pl.DataFrame, col:str) -> dict: 
    return{
        "avg": round(df[col].mean(), 2),
        "med": df[col].median(),
        "min": df[col].min(),
        "max": df[col].max()
    }


def _value_distribution(df:pl.DataFrame, col:str) -> dict:
    counts = df.group_by(col).len()

    return {
        row[col]: row["len"]
    }

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

    total_length_chars = _avg_med_min_max(pdf, "text_characters")
    total_length_words = _avg_med_min_max(pdf, "text_words")
    incoming_length_chars = _avg_med_min_max(
        pdf.filter(
            pl.col("is_incoming") == True
        ),
        "text_characters"
    )
    incoming_length_words = _avg_med_min_max(
        pdf.filter(
            pl.col("is_incoming") == True
        ),
        "text_words"
    )
    outgoing_length_chars = _avg_med_min_max(
        pdf.filter(
            pl.col("is_incoming") == False
        ),
        "text_characters"
    )
    outgoing_length_words = _avg_med_min_max(
        pdf.filter(
            pl.col("is_incoming") == False
        ),
        "text_words"
    )

    logger.info(
        f"Message stats:\
            \nTotal: {number_of_messages}\
            \nTotal length chars: {total_length_chars}\
            \nTotal length words: {total_length_words}\
            \nIncoming: {number_of_incoming}\
            \nIncoming lenth chars: {incoming_length_chars}\
            \nIncoming length words: {incoming_length_words}\
            \nOutgoing: {number_of_outgoing}\
            \nOutgoing length chars: {outgoing_length_chars}\
            \nOutgoing length words: {outgoing_length_words}\
            \nDifference in total and in + out (should = 0): {
                number_of_messages 
                - number_of_incoming
                - number_of_outgoing
            }"
        )
    logger.info("="*20)

    message_stats = {
        "All messages": {
            "Number of messages": number_of_messages,
            "Character stats": total_length_chars,
            "Word stats": total_length_words
        },
        "Incoming messages": {
            "Number of messages": number_of_incoming,
            "Character stats": incoming_length_chars,
            "Word stats": incoming_length_words
        },
        "Outgoing messages": {
            "Number of messages": number_of_outgoing,
            "Character stats": outgoing_length_chars,
            "Word stats": outgoing_length_words
        },
    }

    conversations = pdf.group_by("conversation_code").agg(
        pl.len().alias("total_messages"),
        pl.col("is_incoming").sum().alias("incoming_messages"),
        (~pl.col("is_incoming")).sum().alias("outgoing_messages")
    )

    number_of_conversations = conversations.height
    number_of_messages_conversations = _avg_med_min_max(conversations, "total_messages")
    number_of_incoming_conversations = _avg_med_min_max(conversations, "incoming_messages")
    number_of_outgoing_conversations = _avg_med_min_max(conversations, "outgoing_messages")

    logger.info(
        f"Conversation stats:\
        \nTotal: {number_of_conversations}\
        \nNumber of messages: {number_of_messages_conversations}\
        \nNumber of incoming: {number_of_incoming_conversations}\
        \nNumber of outgoing: {number_of_outgoing_conversations}"
    )
    logger.info("="*20)

    conversation_stats = {
        "Number of conversations": number_of_conversations,
        "All messages stats": number_of_messages_conversations,
        "Incoming messages stats": number_of_incoming_conversations,
        "Outgoing messages stats": number_of_outgoing_conversations
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

    number_of_messages_no_gender = pdf.filter(
        pl.col("gender").is_null()
    ).height

    gender_distribution = {
        "Male": number_of_messages_male,
        "Female": number_of_messages_female,
        "Other": number_of_messages_other,
        "No value": number_of_messages_no_gender
    }

    logger.info(
        f"Gender distribution:\
        \nMale: {number_of_messages_male}\
        \nFemale: {number_of_messages_female}\
        \nOther: {number_of_messages_other}\
        \nNo gender: {number_of_messages_no_gender}\
        \nDifference in total messages and males, females and other (should = no gender): {
            number_of_messages 
            - number_of_messages_male 
            - number_of_messages_female 
            - number_of_messages_other
        }"
    )
    logger.info("="*20)

    number_of_messages_sms = pdf.filter(
        pl.col("requester_channel") == "sms"
    ).height

    number_of_messages_chat = pdf.filter(
        pl.col("requester_channel") == "webchat"
    ).height

    platform_distribution = {
        "SMS": number_of_messages_sms,
        "Chat": number_of_messages_chat
    }

    logger.info(
        f"Platform distribution:\
        \nSMS: {number_of_messages_sms}\
        \nChat: {number_of_messages_chat}\
        \nDifference in total messages and SMS and chat (should = 0): {
            number_of_messages
            - number_of_messages_sms
            - number_of_messages_chat
        }"
    )
    logger.info("="*20)



    stats_dict["Distributions"] = {
        "Age": age_distribution,
        "Gender": gender_distribution,
        "Platform": platform_distribution
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