import argparse
import locale
import os
import re
import sqlite3
from collections import namedtuple
from typing import Optional, List

import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import stop_words
import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from wordcloud import WordCloud

locale.setlocale(locale.LC_ALL, 'it_IT')

PlotInfo = namedtuple('PlotInfo', ['title', 'multimedia', 'aggregation'])
"""Datetype to store plotting information."""

plots: List[PlotInfo] = [
    PlotInfo(title='# Messaggi (Testo)', multimedia=False, aggregation='count'),
    PlotInfo(title='# Messaggi (Contenuti Multimediali)', multimedia=True, aggregation='count'),
    PlotInfo(title='Lunghezza Testo (# Caratteri)', multimedia=False, aggregation='sum'),
    PlotInfo(title='Lunghezza Contenuti Multimediali (# Minuti)', multimedia=True, aggregation='sum')
]
"""Metadata about the plots to do for each data source."""


def load(folder: str, start: Optional[str] = '2021-11', end: Optional[str] = '2025-04') -> pd.DataFrame:
    """Loads the data from the temporary csv file (if given and present) or from the database and the info files.

    :param folder:
        The folder where the data is stored.

    :param start:
        The starting datetime to consider (or None for open interval).

    :param end:
        The ending datetime to consider (or None for open interval).

    :return:
        The raw data about messages.
    """
    # handle time intervals for sql
    end = pd.Timestamp.max if end is None else pd.Timestamp(end)
    start = pd.Timestamp.min if start is None else pd.Timestamp(start)
    # open connection to local database using sqlite3.connect, then build dataframe using pandas.read_sql_query
    # the query consists in creating a temp view which merges messages and chat information (using jid to retrieve
    # sender number in case of group chat) then joining it with jid to obtain sender number in case of one-by-one chat
    # (partially taken from https://github.com/abdulrahmankhayal/WhatsApp-Messages-DB-Analysis-Using-SQL/tree/main)
    data = pd.read_sql_query(
        con=sqlite3.connect(f'{folder}/msgstore.db'),
        sql=f"""
            SELECT jid.user as number,
                   jid_sender.user AS sender,
                   chat.subject as 'group',
                   Strftime('%Y-%m-%d %H:%M:%S', message.timestamp / 1000.0, 'unixepoch') AS datetime,
                   message.from_me as sent,
                   message.text_data AS text,
                   message_media.media_duration as duration,
                   CASE message.message_type
                       WHEN 0 THEN FALSE
                       ELSE TRUE
                   END AS multimedia
            FROM message
            LEFT JOIN chat ON message.chat_row_id = chat._id
            LEFT JOIN jid ON chat.jid_row_id = jid._id
            LEFT JOIN jid AS jid_sender ON message.sender_jid_row_id = jid_sender._id
            LEFT JOIN message_media ON message._id = message_media.message_row_id
            WHERE jid.user <> 'status'
            AND message.timestamp <= {end.timestamp() * 1000.0}
            AND message.timestamp >= {start.timestamp() * 1000.0}
        """
    )
    # convert multimedia duration to minutes and add text length as duration whenever the datatype is not multimedia
    data['duration'] = data.apply(lambda r: r['duration'] / 60.0 if r['multimedia'] else len(r['text']), axis=1)
    data['datetime'] = pd.to_datetime(data['datetime'])
    # replace empty subjects with 'private chat' and remaining subjects with group aliases if present (otherwise delete)
    aliases = pd.read_csv(f'{folder}/groups.csv', index_col='name')['alias'].to_dict()
    data = data[data['group'].isna() | data['group'].isin(aliases)]
    data['group'] = data['group'].map(aliases)
    # when the group is null (private chat) keep the number, otherwise substitute it with the sender
    # (sometimes the sender can be none, meaning that I was the one sending a message in a group chat)
    data['number'] = np.where(data['group'].isna(), data['number'], data['sender'])
    data['number'] = data['number'].map(lambda num: None if num is None else num.split('-')[0]).astype(float)
    # merge the message information with the contacts information then export to a temporary file and return the data
    info = {}
    contacts = pd.read_csv(f'{folder}/contacts.csv')
    # load contacts from raw csv file and iterate each row
    for _, row in contacts.iterrows():
        # remove nan values to correctly manage missing surnames and additional phones
        row = row.dropna()
        name = row.get('First Name', '') + ' ' + row.get('Last Name', '')
        name = name.strip()
        # retrieve numbers (splitting by ' ::: ' keyword) and assign them as keys to the dataset
        # moreover, remove white spaces when present, and add prefix if not present
        for column in ['Phone 1 - Value', 'Phone 2 - Value']:
            phone = row.get(column)
            if phone is None:
                continue
            for number in phone.split(' ::: '):
                number = re.sub(r'[\s\-]', '', number)
                number = number[1:] if number.startswith('+') else f'39{number}'
                info[int(number)] = name
    # merge contacts data with people info (map aliases into "Surname N.")
    people = pd.read_csv(f'{folder}/people.csv', index_col='name')
    contacts = pd.Series(info).to_frame(name='name').reset_index(drop=False, names='number')
    contacts = contacts.join(people, how='inner', on='name')
    contacts['name'] = np.where(contacts['alias'].isna(), contacts['name'], contacts['alias'])
    contacts = contacts.drop(columns='alias')
    # merge contacts data with messages
    data = data.join(other=contacts.set_index('number'), how='inner', on='number')
    data = data.drop(columns=['number', 'sender']).astype({'sent': bool})
    return data[['name', 'individual'] + [c for c in data.columns if c not in ['name', 'individual']]]


def individuals(data: pd.DataFrame, limit: Optional[int] = 20, folder: Optional[str] = None) -> None:
    """Plots a bar chart for the data taken from individual chats.

    :param data:
        The original dataframe.

    :param limit:
        The number of individuals to include in each plot (None to include all of them).

    :param folder:
        The folder where to store the results, or None to show them.
    """
    rows = np.floor(np.sqrt(len(plots))).astype(int)
    cols = np.ceil(len(plots) / rows).astype(int)
    fig = plt.figure(figsize=(60, 60))
    # take private chat data (no group) for the individuals that are considered
    data = data[data['group'].isna() & data['individual'].notna()]
    # for each kind of plot, select the correct datatypes and aggregate on both name and sent
    # (sort using a second group by to collect sum the values of sent and received messages)
    for idx, info in enumerate(plots):
        plt.subplot(rows, cols, idx + 1)
        df = data[data['multimedia'] == info.multimedia]
        df = df.groupby(['individual', 'sent'])['duration'].agg(info.aggregation).reset_index('sent')
        df = df.loc[df.groupby('individual')['duration'].sum().sort_values(ascending=False).head(limit).index]
        ax = fig.gca()
        sns.barplot(
            data=df,
            x='duration',
            y='individual',
            hue=df['sent'].map(lambda sent: 'Inviati' if sent else 'Ricevuti'),
            hue_order=['Ricevuti', 'Inviati'],
            palette=['#0D21A1', '#FFD60A'],
            dodge=True,
            ax=ax
        )
        ax.legend(loc='lower right').set_title('')
        ax.set_xscale('log')
        ax.set_xlim(1, ax.get_xlim()[1])
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_title(info.title, weight='bold', pad=25)
    fig.tight_layout(w_pad=3, h_pad=2)
    if folder is None:
        fig.show()
    else:
        fig.savefig(f'{folder}/individuals.pdf')


def clusters(data: pd.DataFrame, frequency: str = '6MS', folder: Optional[str] = None) -> None:
    """Plots a time series for the data taken from individual chats and grouped according to given clusters.

    :param data:
        The original dataframe.

    :param frequency:
        The aggregation frequency of the time series.

    :param folder:
        The folder where to store the results, or None to show them.
    """
    # take private chat data (no group) for everybody
    data = data[data['group'].isna()]
    # for each kind of plot, select the correct datatypes and aggregate on cluster
    # (sort using a second group by to collect sum the values of sent and received messages)
    fig, axes = plt.subplots(len(plots) + 1, 1, figsize=(40, 60), gridspec_kw={'height_ratios': [1] + [8] * len(plots)})
    handles, labels = [], []
    for ax, info in zip(axes[1:], plots):
        df = data[data['multimedia'] == info.multimedia].set_index('datetime')
        df = df.groupby([pd.Grouper('datetime', freq=frequency), 'cluster'])['duration']
        df = df.agg(info.aggregation).reset_index()
        sns.barplot(
            data=df,
            x='datetime',
            y='duration',
            hue='cluster',
            hue_order=['Famiglia', 'Patria', 'Università', 'Laboratorio', 'Musica', 'Capoeira', 'Precariə'],
            palette=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999'],
            ax=ax
        )
        ax.set_xticks(ax.get_xticks(), labels=df['datetime'].unique().map(lambda s: s.strftime("%b '%y")))
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_yscale('log')
        ax.set_title(info.title, fontweight='bold', pad=25)
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
    axes[0].legend(handles=handles, labels=labels, loc='center', ncols=4, frameon=False)
    axes[0].axis('off')
    fig.tight_layout(w_pad=3, h_pad=2)
    if folder is None:
        fig.show()
    else:
        fig.savefig(f'{folder}/clusters.pdf')


def groups(data: pd.DataFrame, frequency: str = '6MS', folder: Optional[str] = None) -> None:
    """Plots a time series for the data taken from group chats.

    :param data:
        The original dataframe.

    :param frequency:
        The aggregation frequency of the time series.

    :param folder:
        The folder where to store the results, or None to show them.
    """
    # take group chats only
    data = data[data['group'].notna()]
    # for each kind of plot, select the correct datatypes and aggregate on group
    # (sort using a second group by to collect sum the values of sent and received messages)
    fig, axes = plt.subplots(len(plots) + 1, 1, figsize=(40, 60), gridspec_kw={'height_ratios': [1] + [8] * len(plots)})
    handles, labels = [], []
    for ax, info in zip(axes[1:], plots):
        df = data[data['multimedia'] == info.multimedia].set_index('datetime')
        df = df.groupby([pd.Grouper('datetime', freq=frequency), 'group'])['duration']
        df = df.agg(info.aggregation).reset_index()
        sns.barplot(
            data=df,
            x='datetime',
            y='duration',
            hue='group',
            hue_order=['Che Cozza', 'Sleepover Club', 'Farti Phone', 'Bugllismo & Co.', 'Dreamers', 'Asse Precaria'],
            palette=['#dede00', '#e41a1c', '#ff7f00', '#4daf4a', '#984ea3', '#999999'],
            ax=ax
        )
        ax.set_xticks(ax.get_xticks(), labels=df['datetime'].unique().map(lambda s: s.strftime("%b '%y")))
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_yscale('log')
        ax.set_title(info.title, fontweight='bold', pad=25)
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
    axes[0].legend(handles=handles, labels=labels, loc='center', ncols=3, frameon=False)
    axes[0].axis('off')
    fig.tight_layout(w_pad=3, h_pad=2)
    if folder is None:
        fig.show()
    else:
        fig.savefig(f'{folder}/groups.pdf')


def clouds(data: pd.DataFrame, res: str, folder: str) -> None:
    """Creates and stores wordclouds for each contact.

    :param data:
        The original dataframe.

    :param res:
        The path to the resource folder.

    :param folder:
        The folder where to store the results.
    """
    folder = f'{folder}/clouds'
    os.makedirs(folder, exist_ok=True)
    mask = np.array(Image.open(f'{res}/cloud.png'))
    # take private chats and text messages only and convert to tokens using nlp model for pos tagging
    # take model stopwords along with italian and english stopwords, then remove all the punctuation with \W
    nlp = spacy.load('it_core_news_lg')
    tags = {'ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB'}
    stopwords = {*nlp.Defaults.stop_words, *stop_words.get_stop_words('italian'), *stop_words.get_stop_words('english')}
    stopwords = {re.sub(r'\W', '', sw) for sw in stopwords}
    data = data[data['group'].isna() & ~data['multimedia']].groupby('name')['text']
    for i, (name, df) in enumerate(data):
        text = ''
        for msg in tqdm.tqdm(df, desc=f'{i:03}) {name}'):
            text += ' '.join([re.sub(r'\W', '', t.text.lower()) for t in nlp(msg) if t.pos_ in tags]) + ' '
        WordCloud(
            font_path=f'{res}/IndieFlower.ttf',
            stopwords=stopwords,
            collocations=False,
            mask=mask,
            scale=2,
            margin=20,
            max_words=500,
            min_font_size=10,
            contour_width=20,
            min_word_length=2,
            contour_color='black',
            background_color='#FFFFED',
            colormap='viridis'
        ).generate(text).to_file(f'{folder}/{name}.png')


# build argument parser
parser = argparse.ArgumentParser(description='Perform data analysis')
parser.add_argument(
    '-i',
    '--input',
    type=str,
    default='res',
    help='the input folder containing resources'
)
parser.add_argument(
    '-o',
    '--output',
    type=str,
    nargs='?',
    help='the output folder where to store the plots (None to show)'
)

args = parser.parse_args()
sns.set_theme(context='poster', style='whitegrid', font_scale=3)
dataframe = load(folder=args.input)
individuals(dataframe, folder=args.output)
clusters(dataframe, folder=args.output)
groups(dataframe, folder=args.output)
clouds(dataframe, res=args.input, folder='temp' if args.output is None else args.output)
