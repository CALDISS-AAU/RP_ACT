"""Investigate descriptive statistics for cleaned ACT data."""

## IMPORTS ##
# Standard library
import logging
from pathlib import Path
import json

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


def _format_duration(minutes: float) -> str:
    """Format a duration in minutes as 'Xh Ym' or 'Xm'."""

    hours = int(minutes // 60)
    minutes = int(round(minutes % 60))

    if hours == 0:
        return f"{minutes}m"

    return f"{hours}h {minutes}m"


def _avg_med_min_max_time(
    df: pl.DataFrame,
    start_col: str,
    end_col: str,
) -> dict:
    """Calculate average, median, minimum and maximum duration between two datetime columns."""

    durations = (
        df.select(
            (pl.col(end_col) - pl.col(start_col))
            .dt.total_minutes()
            .alias("duration")
        )
        .get_column("duration")
    )

    return {
        "avg": _format_duration(durations.mean()),
        "med": _format_duration(durations.median()),
        "min": _format_duration(durations.min()),
        "max": _format_duration(durations.max()),
    }


def _value_distribution(df: pl.DataFrame, col: str) -> dict:
    counts = df.group_by(col).len()
    total = df.height

    return {
        (row[col] if row[col] is not None else "No value"): {
            "count": row["len"],
            "percentage": round(100 * row["len"] / total, 2)
        }
        for row in counts.iter_rows(named=True)
    }


def _list_value_distribution(df: pl.DataFrame, col: str) -> dict:
    exploded = df.select(col).explode(col)

    counts = exploded.group_by(col).len()
    total_rows = df.height

    return {
        (row[col] if row[col] is not None else "No value"): {
            "count": row["len"],
            "percentage": round(100 * row["len"] / total_rows, 2),
        }
        for row in counts.iter_rows(named=True)
    }


def _list_length_distribution(df: pl.DataFrame, col: str) -> dict:
    lengths = df.with_columns(
        pl.col(col).list.len().alias("num_topics")
    )

    counts = lengths.group_by("num_topics").len().sort("num_topics")
    total_rows = df.height

    return {
        row["num_topics"]: {
            "count": row["len"],
            "percentage": round(100 * row["len"] / total_rows, 2),
        }
        for row in counts.iter_rows(named=True)
    }


def _save_stats(stats: dict, output_path: str | Path, key: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {}

    existing[key] = stats

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4, ensure_ascii=False)
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
        "Conversation_stats (messages pr conversation)": conversation_stats
    }
    logger.info("Stats for messages and conversations have been added to stats_dict")
    logger.info(f"Current stats_dict: {stats_dict}")
    logger.info("="*20)

    # Include distributions
    age_distribution = _value_distribution(pdf, "age_group")
    logger.info(
        f"Age distribution:"
        f"\n{age_distribution}"
    )
    logger.info("=" * 20)

    gender_distribution = _value_distribution(pdf, "gender")
    logger.info(
        f"Gender distribution:"
        f"\n{gender_distribution}"
    )
    logger.info("=" * 20)

    platform_distribution = _value_distribution(pdf, "requester_channel")
    logger.info(
        f"Platform distribution:"
        f"\n{platform_distribution}"
    )
    logger.info("=" * 20)

    time_of_day_distribution = _value_distribution(pdf, "time_of_day")
    logger.info(
        f"Time of day distribution:"
        f"\n{time_of_day_distribution}"
    )
    logger.info("=" * 20)

    season_distribution = _value_distribution(pdf, "season")
    logger.info(
        f"Season distribution:"
        f"\n{season_distribution}"
    )
    logger.info("=" * 20)

    topic_distribution = _list_value_distribution(pdf, "contact_topics")
    logger.info(
        f"Topic distribution:"
        f"\n{topic_distribution}"
    )
    logger.info("=" * 20)

    topic_number_distribution = _list_length_distribution(pdf, "contact_topics")
    logger.info(
        f"Number of topics distribution:"
        f"\n{topic_number_distribution}"
    )
    logger.info("=" * 20)

    stats_dict["Distributions"] = {
        "Age": age_distribution,
        "Gender": gender_distribution,
        "Platform": platform_distribution,
        "Time of day": time_of_day_distribution,
        "Sesaon": season_distribution,
        "Contact topics": topic_distribution,
        "Number of contact topics": topic_number_distribution
    }

    logger.info("Distributions for messages have been added to stats_dict")
    logger.info(f"Current stats_dict: {stats_dict}")
    logger.info("="*20)

    # Saving stats
    _save_stats(stats_dict, output_dir, "Messages")
    logger.info("Stats have been saved to json")
    logger.info("=" * 20)
    logger.info("=" * 20)


def data_investigation_collapsed_messages(
    input_dir: str, 
    output_dir: str,
    logger: logging.Logger,
):
    stats_dict = {}
    pdf = pl.read_parquet(input_dir)

    # Include dataset information
    stats_dict["Dataset_type"] = "Collapsed messages"
    logger.info("Dataset type has been added to stats.")
    logger.info(f"Current stats_dict: {stats_dict}")
    logger.info("="*20)

    # Include message and conversation stats
    number_of_messages = pdf.height
    total_length_chars = _avg_med_min_max(pdf, "text_characters")
    total_length_words = _avg_med_min_max(pdf, "text_words")
    logger.info(
        f"Message stats:\
            \nTotal: {number_of_messages}\
            \nTotal length chars: {total_length_chars}\
            \nTotal length words: {total_length_words}"
        )
    logger.info("="*20)

    message_stats = {
        "All messages": {
            "Number of messages": number_of_messages,
            "Character stats": total_length_chars,
            "Word stats": total_length_words
        },
    }

    conversations = pdf.group_by("conversation_code").agg(
        pl.len().alias("total_messages"),
    )
    number_of_conversations = conversations.height
    number_of_messages_conversations = _avg_med_min_max(conversations, "total_messages")
    logger.info(
        f"Conversation stats:\
        \nTotal: {number_of_conversations}\
        \nNumber of messages: {number_of_messages_conversations}"
    )
    logger.info("="*20)

    conversation_stats = {
        "Number of conversations": number_of_conversations,
        "All messages stats": number_of_messages_conversations,
    }

    stats_dict["stats"] = {
        "Message_stats": message_stats,
        "Conversation_stats (messages pr conversation)": conversation_stats
    }
    logger.info("Stats for messages and conversations have been added to stats_dict")
    logger.info(f"Current stats_dict: {stats_dict}")
    logger.info("="*20)

    # Include distributions
    age_distribution = _value_distribution(pdf, "age_group")
    logger.info(
        f"Age distribution:"
        f"\n{age_distribution}"
    )
    logger.info("=" * 20)

    gender_distribution = _value_distribution(pdf, "gender")
    logger.info(
        f"Gender distribution:"
        f"\n{gender_distribution}"
    )
    logger.info("=" * 20)

    platform_distribution = _value_distribution(pdf, "requester_channel")
    logger.info(
        f"Platform distribution:"
        f"\n{platform_distribution}"
    )
    logger.info("=" * 20)

    time_of_day_distribution = _value_distribution(pdf, "time_of_day")
    logger.info(
        f"Time of day distribution:"
        f"\n{time_of_day_distribution}"
    )
    logger.info("=" * 20)

    season_distribution = _value_distribution(pdf, "season")
    logger.info(
        f"Season distribution:"
        f"\n{season_distribution}"
    )
    logger.info("=" * 20)

    topic_distribution = _list_value_distribution(pdf, "contact_topics")
    logger.info(
        f"Topic distribution:"
        f"\n{topic_distribution}"
    )
    logger.info("=" * 20)

    topic_number_distribution = _list_length_distribution(pdf, "contact_topics")
    logger.info(
        f"Number of topics distribution:"
        f"\n{topic_number_distribution}"
    )
    logger.info("=" * 20)

    stats_dict["Distributions"] = {
        "Age": age_distribution,
        "Gender": gender_distribution,
        "Platform": platform_distribution,
        "Time of day": time_of_day_distribution,
        "Sesaon": season_distribution,
        "Contact topics": topic_distribution,
        "Number of contact topics": topic_number_distribution
    }

    logger.info("Distributions for messages have been added to stats_dict")
    logger.info(f"Current stats_dict: {stats_dict}")
    logger.info("="*20)

    # Saving stats
    _save_stats(stats_dict, output_dir, "Collapsed messages")
    logger.info("Stats have been saved to json")
    logger.info("=" * 20)
    logger.info("=" * 20)


def data_investigation_conversations(
    input_dir: str, 
    output_dir: str,
    logger: logging.Logger,
):
    stats_dict = {}
    pdf = pl.read_parquet(input_dir)

    # Include dataset information
    stats_dict["Dataset_type"] = "Conversations"
    logger.info("Dataset type has been added to stats.")
    logger.info(f"Current stats_dict: {stats_dict}")
    logger.info("="*20)

    # Include conversation stats
    number_of_conversations = pdf.height
    total_length_chars = _avg_med_min_max(pdf, "text_characters")
    total_length_words = _avg_med_min_max(pdf, "text_words")
    logger.info(
        f"Conversation stats:\
            \nTotal: {number_of_conversations}\
            \nTotal length chars: {total_length_chars}\
            \nTotal length words: {total_length_words}"
        )
    logger.info("="*20)

    conversation_duration = _avg_med_min_max_time(
        pdf,
        "msg_created_on",
        "last_contact"
    )
    logger.info(
        f"Conversation duration: {conversation_duration}"
    )
    logger.info("="*20)

    conversation_stats = {
        "All conversations": {
            "Number of conversations": number_of_conversations,
            "Character stats": total_length_chars,
            "Word stats": total_length_words,
            "Duration": conversation_duration
        },
    }

    stats_dict["stats"] = {
        "Conversation_stats": conversation_stats
    }
    logger.info("Stats for conversations have been added to stats_dict")
    logger.info(f"Current stats_dict: {stats_dict}")
    logger.info("="*20)

    # Include distributions
    age_distribution = _value_distribution(pdf, "age_group")
    logger.info(
        f"Age distribution:"
        f"\n{age_distribution}"
    )
    logger.info("=" * 20)

    gender_distribution = _value_distribution(pdf, "gender")
    logger.info(
        f"Gender distribution:"
        f"\n{gender_distribution}"
    )
    logger.info("=" * 20)

    platform_distribution = _value_distribution(pdf, "requester_channel")
    logger.info(
        f"Platform distribution:"
        f"\n{platform_distribution}"
    )
    logger.info("=" * 20)

    time_of_day_distribution = _value_distribution(pdf, "time_of_day")
    logger.info(
        f"Time of day distribution:"
        f"\n{time_of_day_distribution}"
    )
    logger.info("=" * 20)

    season_distribution = _value_distribution(pdf, "season")
    logger.info(
        f"Season distribution:"
        f"\n{season_distribution}"
    )
    logger.info("=" * 20)

    topic_distribution = _list_value_distribution(pdf, "contact_topics")
    logger.info(
        f"Topic distribution:"
        f"\n{topic_distribution}"
    )
    logger.info("=" * 20)

    topic_number_distribution = _list_length_distribution(pdf, "contact_topics")
    logger.info(
        f"Number of topics distribution:"
        f"\n{topic_number_distribution}"
    )
    logger.info("=" * 20)

    stats_dict["Distributions"] = {
        "Age": age_distribution,
        "Gender": gender_distribution,
        "Platform": platform_distribution,
        "Time of day": time_of_day_distribution,
        "Sesaon": season_distribution,
        "Contact topics": topic_distribution,
        "Number of contact topics": topic_number_distribution
    }

    logger.info("Distributions for conversations have been added to stats_dict")
    logger.info(f"Current stats_dict: {stats_dict}")
    logger.info("="*20)

    # Saving stats
    _save_stats(stats_dict, output_dir, "Conversations")
    logger.info("Stats have been saved to json")
    logger.info("=" * 20)
    logger.info("=" * 20)

## __________________ ##