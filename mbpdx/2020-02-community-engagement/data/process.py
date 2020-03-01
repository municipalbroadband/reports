from functools import partial
import os.path

import pandas

HERE = os.path.dirname(__file__)

FORM_DTYPE = pandas.CategoricalDtype(['English Online', 'English Q9NP', 'English Q9P', 'Spanish'])

USE_FREQ_DTYPE = pandas.CategoricalDtype([
    'All day, every day',
    'A few times per day',
    'A few times per week',
    'A few times per month',
    'A few times per year',
    'Never',
])

ISP_DTYPE = pandas.CategoricalDtype([
    'Comcast',
    'CenturyLink',
    'Frontier',
    'A fixed wireless provider (microwave)',
    'Another fiber provider',
    'A dial-up provider',
    'Other',
])

ISP_TECH_DTYPE = pandas.CategoricalDtype([
    'Cable',
    'Fiber',
    'DSL',
    'Fixed wireless',
    'Dial-up',
    'Other',
])

DISLIKE_DTYPE = pandas.CategoricalDtype([
    'Price',
    'Reliability',
    'Customer service',
    'Billing',
    'Lack of choice in providers',
    'Download speed',
    'Upload speed',
    'Bandwidth caps',
    'Lack of network neutrality guarantees',
    'Lack of privacy guarantees',
    "Subscription fees funding providers' lobbying to tilt regulation in their favor",
    'Other',
])

DISPUTE_RESOLVER_DTYPE = pandas.CategoricalDtype([
    'The provider itself',
    'US Congress',
    'Federal Communications Commission',
    'State legislature',
    'Oregon Public Utilities Commission',
    'Oregon Department of Justice',
    'Mount Hood Cable Regulatory Commission',
    'Portland Office of Community Technology',
    'Portland City Council',
    'None of these',
    'Other',
])

IMPORTANCE_DTYPE = pandas.CategoricalDtype([
    'Not important',
    'Less important',
    'Somewhat important',
    'More important',
    'Very important',
], ordered=True)

AGREE_DTYPE = pandas.CategoricalDtype([
    'Strongly disagree',
    'Somewhat disagree',
    'Neither agree nor disagree',
    'Somewhat agree',
    'Strongly agree',
], ordered=True)


def categorize_list(dtype, cell):
    return pandas.Series(cell).astype(dtype).fillna('Other').to_list()


def map_online_dislikes(cell):
    return pandas.Series(cell).replace({
        'Lack of choices in providers': 'Lack of choice in providers',
        'Your subscription fees funding ISPs lobbying to tilt regulation in their favor':
            "Subscription fees funding providers' lobbying to tilt regulation in their favor",
    }).to_list()


def map_online_dispute_resolvers(cell):
    return pandas.Series(cell).replace({
        'Office of Community Technology': 'Portland Office of Community Technology',
        'City Council': 'Portland City Council',
        'None of these institutions can/will help': 'None of these',
    }).to_list()


def map_filter_na(fn, df, col):
    return fn(df[~pandas.isna(df[col])][col])


def map_filter_na_inplace(fn, df, col):
    df.loc[lambda df: ~pandas.isna(df[col]), col] = map_filter_na(fn, df, col)


def combined():
    online = pandas.read_csv(os.path.join(HERE, 'raw-online.csv'))
    online.drop(columns=[
        "Timestamp",
        "In a perfect world, how could your home/business Internet service be better (i.e. things you can't do now)?",
        "What question(s) didn't we ask that we should have, and your answer(s)?",
    ], inplace=True)
    online.rename(columns={
        "Do you use the Internet?": 'internet_use_freq',
        "Do you have Internet in your home/business?": 'has_internet_premise',
        "Who provides your home/business Internet service?": 'isp_raw',
        "About how much is your monthly Internet service in your home/business (excluding mobile plans)?": 'internet_price',
        "Do you ONLY have Internet access through a mobile data plan?": 'has_internet_mobile_only',
        "About how much is the price per month for your family's mobile service?": 'mobile_price',
        "Do you ONLY have Internet access through the generosity of someone who is not in your household (e.g. free wifi, a neighbor, at work, the library)?": 'has_internet_ext_only',
        "What do you dislike about your home/business Internet service? (check all that apply)": 'dislikes',
        "When there is a dispute with your home/business Internet provider, from which institutions can you get satisfaction? (check any that apply)": 'dispute_resolvers',
        "How important is it for students to have Internet access?": 'importance_student',
        "How important is it for low-income families to have Internet access?": 'importance_low_income',
        "Do you support a publicly-owned telecommunications utility?": 'support_utility',
        "How important is user input in governance of a public telecommunications utility?": 'importance_user_input',
        "How important is it that rates pay only for utility costs?": 'importance_rates_direct',
        "Subscribers should subsidize access for families who can't afford home Internet access.": 'subsidize_subscribers',
        "Taxpayers should subsidize access for families who can't afford home Internet access.": 'subsidize_taxpayers',
        "I care enough about this issue that I would financially support a campaign to create a publicly-owned telecommunications utility.": 'support_financial',
    }, inplace=True)
    online['form'] = 'English Online'

    # Remove invalid data.
    online.drop(online[online['internet_use_freq'] == 'Never'].index, inplace=True)

    # Map booleans.
    for col in ['has_internet_premise', 'has_internet_mobile_only', 'has_internet_ext_only']:
        online[col] = online[col] == 'Yes'

    # Map yes/no questions used here to score.
    online['support_utility'] = online['support_utility'].map({'Yes': 5.0, 'No': 1.0})

    # Map frequency onto our dtype.
    online['internet_use_freq'] = online['internet_use_freq'].replace({
        'Within the last day': 'A few times per day',
        'Within the last week': 'A few times per week',
        'Within the last month': 'A few times per month',
        'Within the last year': 'A few times per year',
    })

    # Parse our multiple-choice answers into lists.
    for col in ['dislikes', 'dispute_resolvers']:
        online[col] = online[col].str.split(pat=';')

    # Map dislikes and dispute resolvers onto our dtypes.
    map_filter_na_inplace(lambda s: s.apply(map_online_dislikes), online, 'dislikes')
    map_filter_na_inplace(lambda s: s.apply(map_online_dispute_resolvers), online, 'dispute_resolvers')

    in_person = pandas.read_csv(os.path.join(HERE, 'raw-in-person.csv'))
    in_person.drop(columns=[
        "In a perfect world, how could your internet service be better (what would you like to be able to do that you can't do now)?",
    ], inplace=True)
    in_person.rename(columns={
        'Form': 'form',
        "How often do you use the internet?": 'internet_use_freq',
        "Do you only have internet access through the generosity of someone who is not in your household, such as free Wi-Fi, a neighbor, your workplace, or the library?": 'has_internet_ext_only',
        "Excluding mobile plans, do you have internet access at home?": 'has_internet_premise',
        "Who provides your home internet service?": 'isp_raw',
        "Excluding mobile plans, about how much is the price per month of your home internet service?": 'internet_price',
        "Do you have internet access through a mobile plan?": 'has_internet_mobile',
        "If you marked Yes, about how much is the price per month of your household's mobile plan?": 'mobile_price',
        "What do you dislike about your internet service?": 'dislikes',
        "When you have an issue with your internet provider, from which institutions can you get satisfaction?": 'dispute_resolvers',
        "How important is it for students to have internet access?": 'importance_student',
        "How important is it for low-income families to have internet access?": 'importance_low_income',
        "I support a publicly-owned internet utility.": 'support_utility',
        "Community input is important in governance of an internet utility.": 'importance_user_input',
        "Rates should only pay for utility costs.": 'importance_rates_direct',
        "Subscribers should subsidize internet access for families that can't afford it.": 'subsidize_subscribers',
        "All taxpayers should subsidize internet access for families that can't afford it.": 'subsidize_taxpayers',
        "I would financially support a campaign to create a publicly-owned internet utility.": 'support_financial',
    }, inplace=True)

    # Map booleans.
    for col in ['has_internet_premise', 'has_internet_mobile', 'has_internet_ext_only']:
        in_person[col] = in_person[col] == 'Yes'

    # Detect mobile-only (question originally referred to mobile use in general).
    in_person['has_internet_mobile_only'] = in_person.apply(
        lambda row: not row['has_internet_premise'] and row['has_internet_mobile'],
        axis=1,
    )
    in_person.drop(columns=['has_internet_mobile'], inplace=True)

    # Parse our multiple-choice answers into lists.
    for col in ['dislikes', 'dispute_resolvers']:
        in_person[col] = in_person[col].str.split(pat=', ')

    # Combine datasets.
    data = pandas.concat([online, in_person], ignore_index=True)

    # Remove invalid data.
    data.drop(data[data['has_internet_ext_only'] & (data['has_internet_premise'] | data['has_internet_mobile_only'])].index, inplace=True)
    data.drop(data[data['has_internet_mobile_only'] & data['has_internet_premise']].index, inplace=True)

    # Assign categorical data.
    data['form'] = data['form'].astype(FORM_DTYPE)

    # Extract prices from price columns. Remove any price if other answer
    # precludes it.
    data['internet_price'] = data.apply(
        lambda row: row['internet_price'] if row['has_internet_premise'] else pandas.NA,
        axis=1,
    )
    for col in ['internet_price', 'mobile_price']:
        data[col] = pandas.to_numeric(data[col].str.extract(r'(?:^|\$)(\d+(?:\.\d+)?)(?:$|\b)', expand=False))

    # Map ISP information.
    data['isp'] = map_filter_na(lambda s: s.replace({
        'CenturyLink (fiber)': 'CenturyLink',
        'CenturyLink (DSL)': 'CenturyLink',
        'Frontier (fiber)': 'Frontier',
        'Frontier (DSL)': 'Frontier',
    }).astype(ISP_DTYPE).fillna('Other'), data, 'isp_raw')
    data['isp_tech'] = map_filter_na(lambda s: s.replace({
        'Comcast': 'Cable',
        'CenturyLink (fiber)': 'Fiber',
        'CenturyLink (DSL)': 'DSL',
        'Frontier (fiber)': 'Fiber',
        'Frontier (DSL)': 'DSL',
        'A fixed wireless provider (microwave)': 'Fixed wireless',
        'Another fiber provider': 'Fiber',
        'A dial-up provider': 'Dial-up',
    }).astype(ISP_TECH_DTYPE).fillna('Other'), data, 'isp_raw')
    data.drop(columns=['isp_raw'], inplace=True)

    # Map unordered categories, ignoring NaN.
    mappers = {
        'internet_use_freq': lambda s: s.astype(USE_FREQ_DTYPE),
        'dislikes': lambda s: s.apply(partial(categorize_list, DISLIKE_DTYPE)),
        'dispute_resolvers': lambda s: s.apply(partial(categorize_list, DISPUTE_RESOLVER_DTYPE)),
    }
    for col, mapper in mappers.items():
        map_filter_na_inplace(mapper, data, col)

    # Map ordered categories.
    for col in ['importance_student', 'importance_low_income']:
        data[col] = pandas.Categorical.from_codes(data[col].fillna(0).astype(int)-1, dtype=IMPORTANCE_DTYPE)
    for col in ['support_utility', 'importance_user_input', 'importance_rates_direct', 'subsidize_subscribers', 'subsidize_taxpayers', 'support_financial']:
        data[col] = pandas.Categorical.from_codes(data[col].fillna(0).astype(int)-1, dtype=AGREE_DTYPE)

    return data


if __name__ == '__main__':
    combined().to_json(path_or_buf='normalized.json', orient='records', indent=4)
