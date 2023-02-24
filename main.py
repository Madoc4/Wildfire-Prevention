from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as px


def get_probability(curr_data):
    # Data Formatting
    data = pd.read_csv('data.csv')
    data = data.fillna(0)

    # Splits the data into a training set and a test set
    x = data[['Temperature (F)', 'Humidity (%)', 'Precipitation (in)', 'Wind Speed (mph)', 'Pressure (inHg)']]
    y = data['Wildfire']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Trains the model on the training set
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Predict the probability of a wildfire occurring for a particular location
    temperature = curr_data['temperature']
    humidity = curr_data['humidity']
    precipitation = curr_data['precipitation']
    wind_speed = curr_data['wind_speed']
    pressure = curr_data['pressure']

    x_new = [[temperature, humidity, precipitation, wind_speed, pressure]]
    x_new = pd.DataFrame(x_new, columns=x.columns)
    probability = model.predict_proba(x_new)[0][1] * 100

    return f'{probability:.2f}'


def get_data(zip):
    # Construct the URL for the weather website
    url = f'https://weather.com/weather/today/l/{zip}'

    # Make an HTTP request to the website using the requests library
    page = requests.get(url)

    # Parse the HTML response using BeautifulSoup
    soup = BeautifulSoup(page.content, 'html.parser')

    # Scrape information from the website
    items = [item.get_text(strip=True) for item in soup.find_all(class_='WeatherDetailsListItem--wxData--kK35q')]
    loc_name = soup.find(class_='CurrentConditions--location--1YWj_').get_text(strip=True)
    city = (loc_name[0].lower()).strip()
    state = (loc_name[1].lower()).strip()

    # Get data from list
    wind_speed = items[1].split('n')[-1].split(' ')[0]
    humidity = items[2].split('%')[0]
    # Placeholder value for pressure
    pressure = 29.92
    if 'Arrow Up' in items[4]:
        pressure = items[4].split('p')[1].split(' ')[0]
    elif 'Arrow Down' in items[4]:
        pressure = items[4].split('n')[1].split(' ')[0]
    elif 'Arrow Left' in items[4] or 'Arrow Right' in items[4]:
        pressure = items[4].split('t')[1].split(' ')[0]
    temperature = soup.find('div', class_='CurrentConditions--primary--2DOqs').get_text().split("°")[0]

    # Does the same process on above, but on a different website to find precipitation
    url = f'https://www.wunderground.com/forecast/us/{state}/{city}'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    new_items = [item.get_text(strip=True) for item in soup.find_all(class_='wu-value wu-value-to')]
    precipitation = new_items[-1]

    # Return the scraped data as a dictionary
    return {'temperature': float(temperature), 'humidity': float(humidity), 'wind_speed': float(wind_speed),
            'pressure': float(pressure), 'precipitation': float(precipitation)}


def get_accuracy():
    data = pd.read_csv('data.csv')
    data = data.fillna(0)
    # Splits the data into a training set and a test set
    x = data[['Temperature (F)', 'Humidity (%)', 'Precipitation (in)', 'Wind Speed (mph)', 'Pressure (inHg)']]
    y = data['Wildfire']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Calculate the accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1: {f1:.2f}')


def plot_average(probabilities):
    # Reformats the dictionary into a dataframe
    df = pd.DataFrame()
    df['State'] = [key for key in probabilities.keys()]
    df['Percentage Chance of Wildfire'] = [value for value in probabilities.values()]
    # Maps the state names to their corresponding two letter codes
    state_codes = {'Alabama': 'AL',
                   'Alaska': 'AK',
                   'Arizona': 'AZ',
                   'Arkansas': 'AR',
                   'California': 'CA',
                   'Colorado': 'CO',
                   'Connecticut': 'CT',
                   'Delaware': 'DE',
                   'District of Columbia': 'DC',
                   'Florida': 'FL',
                   'Georgia': 'GA',
                   'Hawaii': 'HI',
                   'Idaho': 'ID',
                   'Illinois': 'IL',
                   'Indiana': 'IN',
                   'Iowa': 'IA',
                   'Kansas': 'KS',
                   'Kentucky': 'KY',
                   'Louisiana': 'LA',
                   'Maine': 'ME',
                   'Maryland': 'MD',
                   'Massachusetts': 'MA',
                   'Michigan': 'MI',
                   'Minnesota': 'MN',
                   'Mississippi': 'MS',
                   'Missouri': 'MO',
                   'Montana': 'MT',
                   'Nebraska': 'NE',
                   'Nevada': 'NV',
                   'New Hampshire': 'NH',
                   'New Jersey': 'NJ',
                   'New Mexico': 'NM',
                   'New York': 'NY',
                   'North Carolina': 'NC',
                   'North Dakota': 'ND',
                   'Ohio': 'OH',
                   'Oklahoma': 'OK',
                   'Oregon': 'OR',
                   'Pennsylvania': 'PA',
                   'Rhode Island': 'RI',
                   'South Carolina': 'SC',
                   'South Dakota': 'SD',
                   'Tennessee': 'TN',
                   'Texas': 'TX',
                   'Utah': 'UT',
                   'Vermont': 'VT',
                   'Virginia': 'VA',
                   'Washington': 'WA',
                   'West Virginia': 'WV',
                   'Wisconsin': 'WI',
                   'Wyoming': 'WY'
                   }
    df['State'] = df['State'].map(state_codes)
    df = df.rename(columns={'State': 'code'})

    # Creates a map of the US and colors the states with their corresponding wildfire probability
    fig = px.Figure(data=px.Choropleth(locations=df['code'], z=df['Percentage Chance of Wildfire'],
                                       locationmode='USA-states', colorscale='Reds', colorbar_title='Chance of Wildfire'
                                       , zmin=0, zmax=100, zmid=50))
    fig.update_layout(title_text='Average Chance of Wildfire by State in Percentage', geo_scope='usa')

    fig.show()


if __name__ == '__main__':
    # Dictionary of three zip codes per state in mainland US arranged in alphabetical order.

    '''TO DO: Give bigger states more zip_codes so it's more accurate
    https://zipmap.net/'''

    zip_codes = {'Alabama': [35674, 36264, 36502],
                 'Alaska': [99654, 99740, 99580],
                 'Arizona': [86411, 85546, 85645],
                 'Arkansas': [72738, 71758, 72442],
                 'California': [92274, 96048, 95920],
                 'Colorado': [80435, 81073, 80517],
                 'Connecticut': ['06473', '06255', '06068'],
                 'Delaware': [21601, 21659, 21915],
                 'District of Columbia': [20001, 20507, 20237],
                 'Florida': [33440, 32640, 32443],
                 'Georgia': [31071, 30736, 30673],
                 'Hawaii': [96725, 96790, 96789],
                 'Idaho': [83263, 83604, 83810],
                 'Illinois': [62271, 62411, 61048],
                 'Indiana': [46574, 47959, 47452],
                 'Iowa': [50436, 51579, 50025],
                 'Kansas': [67732, 66733, 67579],
                 'Kentucky': [41339, 42726, 42602],
                 'Louisiana': [71064, 71357, 70548],
                 'Maine': ['04783', '04945', '04010'],
                 'Maryland': [21550, 21530, 21562],
                 'Massachusetts': ['01230', '01010', '01827'],
                 'Michigan': [49920, 48661, 49247],
                 'Minnesota': [56164, 56627, 56345],
                 'Mississippi': [38654, 39465, 38756],
                 'Missouri': [64831, 63434, 65340],
                 'Montana': [59846, 59332, 59418],
                 'Nebraska': [69339, 68355, 69157],
                 'Nevada': [89825, 89404, 89046],
                 'New Hampshire': ['03470', '03835', '03592'],
                 'New Jersey': ['08015', '08205', '08043'],
                 'New Mexico': [88020, 88260, 87581],
                 'New York': [10001, 14086, 11735],
                 'North Carolina': [28337, 27839, 28747],
                 'North Dakota': [58220, 58634, 58542],
                 'Ohio': [44102, 43062, 44601],
                 'Oklahoma': [73112, 74027, 74501],
                 'Oregon': [97212, 97914, 97838],
                 'Pennsylvania': [19102, 16657, 17701],
                 'Rhode Island': ['02864', '02814', '02921'],
                 'South Carolina': [29401, 29689, 29936],
                 'South Dakota': [57101, 57790, 57501],
                 'Tennessee': [37201, 38472, 38548],
                 'Texas': [78201, 79007, 79821],
                 'Utah': [84201, 84764, 84501],
                 'Vermont': ['05450', '05907', '05343'],
                 'Virginia': [23219, 24651, 24557],
                 'Washington': [98101, 99362, 99301],
                 'West Virginia': [25301, 26801, 26143],
                 'Wisconsin': [53201, 54481, 54601],
                 'Wyoming': [82001, 82223, 82334]}

    proba_by_state = {}
    # Runs through the dictionary of states and their zipcodes
    count = 1
    for state, zipcodes in zip_codes.items():
        avg_proba = 0
        # Gets average wildfire probability by state
        for zip in zipcodes:
            print(f'{count}. {zip}: {state}')
            count += 1
            avg_proba += float(get_probability(get_data(zip)))
        proba_by_state[state] = f'{avg_proba / len(zip_codes[state]): .2f}'
    # Calls the plotting function
    plot_average(proba_by_state)
