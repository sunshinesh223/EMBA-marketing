
companys = [{"company_name":"colorline", "review_website":"https://no.trustpilot.com/review/www.colorline.no?languages=all"},
            {"company_name":"AIDA", "review_website":"https://no.trustpilot.com/review/www.aida.de?languages=all"},
            {"company_name":"DFDS", "review_website":"https://no.trustpilot.com/review/dfds.com?languages=all"},
            {"company_name":"Stena Line","review_website":"https://no.trustpilot.com/review/stenaline.com?languages=all"}]

base_filepath = "./Data"

#=======================================================================================================================
# Checking the data
#=======================================================================================================================
base_filepath = "/Users/jhc/personal/mba/marketing/individual_assignment/trustpilot/data1"
deliveroo_filepath = os.path.join(base_filepath, "deliveroo_trustpilot_1000_reviews_raw_data_raw_1.pkl")
ocado_filepath = os.path.join(base_filepath, "ocado_trustpilot_1000_reviews_raw_data_raw_1.pkl")
sainsburys_filepath = os.path.join(base_filepath, "sainsburys_trustpilot_1000_reviews_raw_data_raw_1.pkl")
aldi_filepath = os.path.join(base_filepath, "aldi_trustpilot_1000_reviews_raw_data_raw_1.pkl")
getir_filepath = os.path.join(base_filepath, "getir_trustpilot_1000_reviews_raw_data_raw_1.pkl")
tesco_filepath = os.path.join(base_filepath, "tesco_trustpilot_1000_reviews_raw_data_raw_1.pkl")
waitrose_filepath = os.path.join(base_filepath, "waitrose_trustpilot_1000_reviews_raw_data_raw_1.pkl")
gousto_filepath = os.path.join(base_filepath, "gousto_trustpilot_1000_reviews_raw_data_raw_1.pkl")
just_eat_filepath = os.path.join(base_filepath, "just_eat_trustpilot_1000_reviews_raw_data_raw_1.pkl")
morrisons_filepath = os.path.join(base_filepath, "morrisons_trustpilot_1000_reviews_raw_data_raw_1.pkl")

dfd = pd.read_pickle(deliveroo_filepath)
dfo = pd.read_pickle(ocado_filepath)
dfs = pd.read_pickle(sainsburys_filepath)
dfa = pd.read_pickle(aldi_filepath)
dfg = pd.read_pickle(getir_filepath)
dft = pd.read_pickle(tesco_filepath)
dfw = pd.read_pickle(waitrose_filepath)
dfgou = pd.read_pickle(gousto_filepath)
dfj = pd.read_pickle(just_eat_filepath)
dfm = pd.read_pickle(morrisons_filepath)


secret_name = "<add here>"
secret_key = "<add here>"
organization_id = "<add here>"


#=======================================================================================================================
# Categorising the Data
#=======================================================================================================================
import os
import openai
import pandas as pd

openai.organization = organization_id
openai.api_key = secret_key
#openai.Model.list()

#model = "davinci-002"
model = "gpt-4"

# text = '\nsearch for the key to effective motivation has served as the ‘holy grail’ of management theory; \n'
# content = "can you summarize the following text: {}".format(text)
# completion = openai.ChatCompletion.create(
#     model=model,
#     temperature = 0.0,
#     messages=[{"role": "user", "content": content}],
# )
# reply_content = completion.choices[0].message.content
# print(reply_content)

#=======================================================================================================================

reviews_df = dfd
reviews_df = dfo
reviews_df = dfs
reviews_df = dfa
reviews_df = dfg
reviews_df = dft


## Set your OpenAI API key
#openai.organization = 'your_organization_id'
#openai.api_key = 'your_secret_key'

# Load your DataFrame (assuming it's already loaded, we start from here)
# reviews_df = pd.read_excel('path_to_your_excel_file.xlsx')

# Generate categories based on a sample of reviews
sample_reviews = reviews_df['body_text'].sample(n=50).tolist()
sample_text = "\n\n".join(sample_reviews)

generate_categories_prompt = f"Based on the following reviews, suggest appropriate categories to classify them:\n\n{sample_text}"

category_completion = openai.ChatCompletion.create(
    model=model,
    messages=[{"role": "user", "content": generate_categories_prompt}],
)

categories = category_completion.choices[0].message.content.strip()
print("Suggested categories:\n", categories)



"""

company_name = "ocado"
company_name = "sainsburys"
#company_name = "deliveroo"
filepath_pickle1 = "{}/{}_trustpilot_{}_reviews_{}_raw_1.pkl".format(base_filepath, company_name, num_reviews, "raw_data")
df = pd.read_pickle(filepath_pickle1)

Ocado:
Suggested categories:
1. Delivery Services
2. Products Quality
3. Customer Service
4. Pricing
5. Ocado App/Website
6. Packaging
7. Discount Offers
8. Stock Availability
9. Communication/ Information Transparency
10. Employee Behavior/ Driver Treatment
11. Overcharging/Double Charging
12. Substitution Policy
13. Refund/Return Process
14. Food Hygiene
15. Delivery Cancellations/Missing Items
16. Product Expiration Dates
17. Payment Issues
18. Loyalty Programs/Smart Pass Experience
19. Value for Money
20. Diversity in Product Range.


Sainsburys:

Suggested categories:
1. Website and App Interface 
2. Online Shopping Experience 
3. Product Quality and Availability 
4. Delivery Service 
5. Substitution Policy 
6. Click & Collect Service 
7. Online Pricing and Sales 
8. Customer Service 
9. Store Experience 
10. Payment and Refund Issues.

Suggested categories:
 1. Online Shopping Experience
2. Delivery Service
3. Store Experience
4. Product Quality
5. Customer Service
6. Click & Collect Service
7. Pricing and Offers
8. Store Layout and Navigation 
9. Loyalty Program (Nectar Points)
10. Complaint Handling
11. Substitutions and Availability of Products
12. Parking Facilities
13. Disability Access.


Deliveroo:

Suggested categories:
1. Poor Customer Service
2. Food Quality and Temperature Issues
3. Inaccurate Delivery
4. Complaints of Theft
5. Unethical Practices
6. Terrible Refund Policy
7. Unprofessional Drivers
8. Unsatisfactory Resolution
9. Late Delivery 
10. Missing Items in the Order 
11. False Advertising/Hidden Charges
12. App Functionality Issues
13. Racism 
14. Overpricing
15. Health Safety Hazard
16. Billing Issues.

Suggested categories:
 1. Bad food quality
2. Delivery Issues
3. Poor customer service
4. Scam concerns
5. Dishonest practices
6. Refund Problems
7. Price discrepancies
8. Driver Misconduct
9. Language Barriers
10. Late Delivery
11. Wrong Order Delivered
12. Stolen Items
13. Cold food
14. Missing Items
15. Order Cancellation Issues
16. Accessibility and Communication Issues.

Aldi

Suggested categories:
 1. Food Quality and Freshness
2. Pricing Discrepancies and Discount Problems
3. Customer Service and Staff Behavior
4. Product Misrepresentation
5. Poor Store Hygiene and Organization
6. Discriminatory Treatment and Inappropriate Policies
7. Trust and Honesty Issues
8. Poor Online Shopping Experience
9. Faulty Products and Delayed Refunds
10. Overpricing and Affordability Issues
11. Mismanagement and Insufficient Staff Training
12. Ineffectiveness of Customer Complaint Responses
13. Negative Personal Experiences 
14. Health and Safety Concerns
15. Technical Problems with Online Platform
16. Store Environment and Temperature
17. Disability Discrimination and Lack of Support
18. Policy and Regulatory Violations
19. Poor Delivery Service and Delays
20. Privacy Issues and Inappropriate Bag Checking. 
21. Favouritism and Inconsistent Rules Implementation
22. Negative Impact of Self-Checkout System on Shopping Experience
23. False Advertising and Error in Labelling
24. Bad Exchange and Refund Policies 
25. Customers feeling harassed or accused of theft.


Getir

Suggested categories:
1. Delivery Speed
2. Customer Service
3. Product Quality
4. Promotions/Offers
5. Packaging and Handling
6. App Functionality
7. Accuracy of Order
8. Value for Money
9. Delivery Area Coverage
10. Driver Behaviour
11. Company Ethics and Practices
12. Product Variety. 
13. Refund/Return Policy. 
14. Employment Experiences
15. Company Longevity Predictions
16. User Interface & Experience.

Tesco:

Suggested categories:
1. Poor Store Condition
2. Misleading Pricing 
3. Problem with Delivery Service
4. Issue with Online Shopping
5. Difficulty with Customer Communication
6. Switching from Other Store
7. High Quality of Product
8. Misleading Advertisement
9. Issue with Stock Availability
10. Pricing Complaint
11. Poor Staff Behavior
12. Difficulty with Technology
13. Positive Store Experience
14. Negative Incident in Store
15. Mismanaged Contract
16. Difficulty Using Vouchers
17. False Price Labeling
18. Problem with Parking
19. Price Inflation
20. Difficulty with Redemption of Voucher
21. Rude Staff
22. Negative Store Environment 
23. Unfair productService
24. Poor Customer Service
25. Inefficiency in Customer Service
26. Substandard Product Delivery
27. Difficulty with Refunds
28. Misleading Marketing
29. Problem with Store Policy 
30. Rude Delivery Driver.
31. Positive Staff Behavior
32. Poor in-store Service
33. Issue with Product Quality.
34. Cancellation of Service.
35. Problems with Store's Security Practices.
36. Problem with Product Pricing. 
37. Problem with Product Expiry Date.
38. Mix-up in Product Delivery. 
39. Problem with Product Availability.
40. Store Causes Deceit.
41. Problem with Store Time Management.
42. Negative Product Review.
43. Late Order Delivery.
44. Strict Store Security.
45. Friendly Staff.
46. Problem with Store's Access Control.
47. Store Overcharging. 
48. Positive Shopping Experience at the Store. 
49. Store's Ignorance to Store Policy.
50. Unpleasant Taste of Product.
51. Erroneous Ban from Store.
52. Appreciation for Manned Tills.
53. Product Too Salty.
"""

"""
Consolidated list:
"""

import os
import openai
import pandas as pd

openai.organization = organization_id
openai.api_key = secret_key
model = "gpt-4"

categories = \
"""
1. Delivery Services
2. Product Quality
3. Customer Service
4. Pricing and Offers
5. Online Shopping Experience
6. Website and App Interface
7. Packaging and Handling
8. Stock Availability
9. Communication/Information Transparency
10. Employee/Driver Behavior
11. Overcharging/Double Charging
12. Substitution Policy
13. Refund/Return Process
14. Food Hygiene and Safety
15. Delivery Cancellations/Missing Items
16. Product Expiration Dates
17. Payment and Billing Issues
18. Loyalty Programs
19. Value for Money
20. Product Range Diversity
21. Click & Collect Service
22. Store Experience and Layout
23. Parking and Disability Access
24. Complaint Handling
25. Misleading Pricing and Advertisement
26. Technical Problems with Online Platform
27. Store Hygiene and Organization
28. Store Environment and Temperature
29. Policy and Regulatory Violations
30. Unfair Product/Service Practices
"""

def categorize_review(review, categories):
    content = f"Based on the following categories:\n{categories}\n\nCan you state if this review was Overall: 'Positive' or 'Negative', and would the reviewer 'Use again' or 'Not use again', also Please categorize the following review into one of the given categories:\n\n{review}"

    print(content)
    print()
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.0,
            messages=[{"role": "user", "content": content}],
        )
        reply_content = completion.choices[0].message.content.strip()
        return reply_content
    except Exception as e:
        return f"Error: {str(e)}"


reviews_df = dfj
num_reviews = 550
reviews_df2 = reviews_df.iloc[:num_reviews]
print(categories)
all_review_categories = []
for i in range(num_reviews):
    print(f"Review {i}")
    print(reviews_df2['body_text'][i])
    review_categories = categorize_review(reviews_df2['body_text'][i], categories)
    print(f"Review {i}\nCategories: {review_categories}")
    all_review_categories.append(review_categories)
    time.sleep(1)
    print()
    if i % 100 == 99:
        print("Extra sleep")
        time.sleep(60)

# Processing the review results
import pandas as pd
import re

company_name = "aldi"
company_name_title = "Aldi"


def extract_information(text):

    overall_match = re.search(r"Overall: '?(Positive|Negative)'?", text)
    use_again_match = re.search(r"Use again: '?(Yes|No|Not use again|Use again)'?", text, re.IGNORECASE)
    categories_match = re.search(r"(Category|Categories):([\s\S]+)", text)

    overall = overall_match.group(1) if overall_match else None
    use_again = use_again_match.group(1) if use_again_match else None
    if use_again:
        use_again = use_again.lower().replace('use again', 'yes').replace('not use again', 'no').capitalize()

    if categories_match:
        categories = categories_match.group(2)
        categories = categories.strip()
        categories = re.sub(r"\d+\.\s*", "", categories).replace('\n', ' ')
        categories_list = [cat.strip() for cat in categories.split(',')]
        categories_list = [cat.strip("'") for cat in categories_list]
        categories_list = [cat.strip('"') for cat in categories_list]
        categories_list = [cat.strip('.') for cat in categories_list]
    else:
        categories_list = []

    return text, overall, use_again, categories_list

data = all_review_categories
records = [extract_information(item) for item in data]
df_processed = pd.DataFrame(records, columns=["raw_text", "Overall", "Use again", "Category"])
df2 = pd.concat([reviews_df, df_processed], axis=1)

filepath_excel1 = "{}/{}_trustpilot_{}_reviews_{}_processed_categories_3.xlsx".format(base_filepath, company_name, num_reviews, "raw_data")
df2.to_excel(filepath_excel1, index=False)
print(filepath_excel1)

filepath_pickle1 = "{}/{}_trustpilot_{}_reviews_{}_processed_categories_3.pickle".format(base_filepath, company_name, num_reviews, "raw_data")
df2.to_pickle(filepath_pickle1)
print(filepath_pickle1)

df3 = df2[:500]
df = df3
df.columns = df.columns.str.lower().str.replace(' ', '_')
are_all_lists = df['category'].apply(lambda x: isinstance(x, list)).all()
non_list_rows = df[~df['category'].apply(lambda x: isinstance(x, list))]
print(non_list_rows)
df['num_categories'] = df['category'].apply(len)


valid_categories = [
    'Delivery Services', 'Product Quality', 'Customer Service', 'Pricing and Offers',
    'Online Shopping Experience', 'Website and App Interface', 'Packaging and Handling',
    'Stock Availability', 'Communication/Information Transparency', 'Employee/Driver Behavior',
    'Overcharging/Double Charging', 'Substitution Policy', 'Refund/Return Process',
    'Food Hygiene and Safety', 'Delivery Cancellations/Missing Items', 'Product Expiration Dates',
    'Payment and Billing Issues', 'Loyalty Programs', 'Value for Money', 'Product Range Diversity',
    'Click & Collect Service', 'Store Experience and Layout', 'Parking and Disability Access',
    'Complaint Handling', 'Misleading Pricing and Advertisement', 'Technical Problems with Online Platform',
    'Store Hygiene and Organization', 'Store Environment and Temperature', 'Policy and Regulatory Violations',
    'Unfair Product/Service Practices'
]

def split_into_valid_categories(input_string, valid_categories):
    matched_categories = []
    for category in valid_categories:
        if category in input_string:
            matched_categories.append(category)
            # Remove the found category from the input string to avoid duplicate matches
            input_string = input_string.replace(category, '')
    return matched_categories

def split_and_clean_categories(entry):
    if isinstance(entry, list):
        entry = [cat.strip() for cat in entry]
        entry = [cat.strip('"') for cat in entry]
        entry = [cat.strip("'") for cat in entry]
        new_entry = []
        for item in entry:
            if item not in valid_categories:
                if "' and '" in item:
                    new_entry.extend([cat.strip() for cat in item.split("' and '")])
                else:
                    new_entry.extend(split_into_valid_categories(item, valid_categories))
                    #print(f"unrecognised 1: '{item}', {type(item)}")
                    #import pdb;pdb.set_trace()
            else:
                new_entry.append(item)
        entry = new_entry
        for item in entry:
            if item not in valid_categories:
                print(f"unrecognised 2: '{item}'")
        return entry
    elif isinstance(entry, str) and "' and '" in entry:
        return [cat.strip() for cat in entry.split("' and '")]
    else:
        print(f"unrecognised: '{entry}'")

df['category2'] = df['category'].apply(split_and_clean_categories)
df['num_categories2'] = df['category2'].apply(len)

for category in valid_categories:
    category_name = "cat_" + category.lower().replace(' ', '_')
    df[category_name] = df['category2'].apply(lambda x: 1 if category in x else 0)

cat_columns = [col for col in df.columns if col.startswith('cat_')]
df[cat_columns].sum()
df.groupby(['overall', 'use_again'])[cat_columns].sum()
df.groupby(['overall', 'use_again', 'rating_raw'])[cat_columns].sum()

filepath_excel1 = "{}/{}_trustpilot_{}_reviews_{}_exploded_categories_4.xlsx".format(base_filepath, company_name, num_reviews, "raw_data")
df.to_excel(filepath_excel1, index=False)
print(filepath_excel1)

filepath_pickle1 = "{}/{}_trustpilot_{}_reviews_{}_exploded_categories_4.pickle".format(base_filepath, company_name, num_reviews, "raw_data")
df.to_pickle(filepath_pickle1)
print(filepath_pickle1)

"""
Deliveroo:

cat_delivery_services                          240
cat_product_quality                             18
cat_customer_service                           241
cat_pricing_and_offers                          34
cat_online_shopping_experience                   4
cat_website_and_app_interface                    4
cat_packaging_and_handling                       7
cat_stock_availability                           9
cat_communication/information_transparency      47
cat_employee/driver_behavior                    72
cat_overcharging/double_charging                40
cat_substitution_policy                          6
cat_refund/return_process                      143
cat_food_hygiene_and_safety                     35
cat_delivery_cancellations/missing_items       152
cat_product_expiration_dates                     0
cat_payment_and_billing_issues                  27
cat_loyalty_programs                             9
cat_value_for_money                              7
cat_product_range_diversity                      1
cat_click_&_collect_service                      0
cat_store_experience_and_layout                  0
cat_parking_and_disability_access                1
cat_complaint_handling                         106
cat_misleading_pricing_and_advertisement        15
cat_technical_problems_with_online_platform     13
cat_store_hygiene_and_organization               0
cat_store_environment_and_temperature            0
cat_policy_and_regulatory_violations            19
cat_unfair_product/service_practices            46

df.groupby(['overall', 'use_again'])[cat_columns].sum()
                    cat_delivery_services  cat_product_quality  cat_customer_service  cat_pricing_and_offers  cat_online_shopping_experience  ...  cat_technical_problems_with_online_platform  cat_store_hygiene_and_organization  cat_store_environment_and_temperature  cat_policy_and_regulatory_violations  cat_unfair_product/service_practices
overall  use_again                                                                                                                            ...                                                                                                                                                                                                    
Negative No                           231                   17                   240                      34                               3  ...                                           13                                   0                                      0                                    19                                    46
         Yes                            0                    0                     0                       0                               0  ...                                            0                                   0                                      0                                     0                                     0
Positive Yes                            9                    1                     1                       0                               1  ...                                            0                                   0                                      0                                     0                                     0

[3 rows x 30 columns]

df.groupby(['overall', 'use_again', 'rating_raw'])[cat_columns].sum()

                                           cat_delivery_services  cat_product_quality  cat_customer_service  cat_pricing_and_offers  cat_online_shopping_experience  ...  cat_technical_problems_with_online_platform  cat_store_hygiene_and_organization  cat_store_environment_and_temperature  cat_policy_and_regulatory_violations  cat_unfair_product/service_practices
overall  use_again rating_raw                                                                                                                                        ...                                                                                                                                                                                                    
Negative No        Rated 1 out of 5 stars                    221                   16                   239                      32                               3  ...                                           12                                   0                                      0                                    19                                    44
                   Rated 2 out of 5 stars                      9                    1                     1                       2                               0  ...                                            1                                   0                                      0                                     0                                     2
                   Rated 3 out of 5 stars                      1                    0                     0                       0                               0  ...                                            0                                   0                                      0                                     0                                     0
         Yes       Rated 5 out of 5 stars                      0                    0                     0                       0                               0  ...                                            0                                   0                                      0                                     0                                     0
Positive Yes       Rated 3 out of 5 stars                      1                    0                     0                       0                               0  ...                                            0                                   0                                      0                                     0                                     0
                   Rated 5 out of 5 stars                      8                    1                     1                       0                               1  ...                                            0                                   0                                      0                                     0                                     0

[6 rows x 30 columns]

Ocado:

cat_delivery_services                          223
cat_product_quality                            109
cat_customer_service                           195
cat_pricing_and_offers                          70
cat_online_shopping_experience                  36
cat_website_and_app_interface                   30
cat_packaging_and_handling                      60
cat_stock_availability                          96
cat_communication/information_transparency     106
cat_employee/driver_behavior                   148
cat_overcharging/double_charging                43
cat_substitution_policy                         54
cat_refund/return_process                       48
cat_food_hygiene_and_safety                      9
cat_delivery_cancellations/missing_items       105
cat_product_expiration_dates                    43
cat_payment_and_billing_issues                  58
cat_loyalty_programs                            18
cat_value_for_money                             49
cat_product_range_diversity                     50
cat_click_&_collect_service                      1
cat_store_experience_and_layout                  0
cat_parking_and_disability_access                5
cat_complaint_handling                          67
cat_misleading_pricing_and_advertisement        18
cat_technical_problems_with_online_platform     21
cat_store_hygiene_and_organization               2
cat_store_environment_and_temperature            0
cat_policy_and_regulatory_violations            17
cat_unfair_product/service_practices            33
dtype: int64

In [229]: df.groupby(['overall', 'use_again'])[cat_columns].sum()
Out[229]: 
                    cat_delivery_services  cat_product_quality  cat_customer_service  cat_pricing_and_offers  cat_online_shopping_experience  ...  cat_technical_problems_with_online_platform  cat_store_hygiene_and_organization  cat_store_environment_and_temperature  cat_policy_and_regulatory_violations  cat_unfair_product/service_practices
overall  use_again                                                                                                                            ...                                                                                                                                                                                                    
Negative No                           133                   45                   142                      32                               7  ...                                           18                                   2                                      0                                    17                                    31
         Yes                            2                    0                     2                       0                               0  ...                                            0                                   0                                      0                                     0                                     1
Positive Yes                           88                   62                    50                      37                              29  ...                                            3                                   0                                      0                                     0                                     1

[3 rows x 30 columns]

In [230]: df.groupby(['overall', 'use_again', 'rating_raw'])[cat_columns].sum()
Out[230]: 
                                           cat_delivery_services  cat_product_quality  cat_customer_service  cat_pricing_and_offers  cat_online_shopping_experience  ...  cat_technical_problems_with_online_platform  cat_store_hygiene_and_organization  cat_store_environment_and_temperature  cat_policy_and_regulatory_violations  cat_unfair_product/service_practices
overall  use_again rating_raw                                                                                                                                        ...                                                                                                                                                                                                    
Negative No        Rated 1 out of 5 stars                    111                   24                   118                      22                               3  ...                                           13                                   1                                      0                                    15                                    26
                   Rated 2 out of 5 stars                     17                   15                    19                       7                               4  ...                                            2                                   1                                      0                                     1                                     3
                   Rated 3 out of 5 stars                      4                    6                     4                       3                               0  ...                                            2                                   0                                      0                                     1                                     2
                   Rated 4 out of 5 stars                      1                    0                     1                       0                               0  ...                                            1                                   0                                      0                                     0                                     0
         Yes       Rated 1 out of 5 stars                      1                    0                     0                       0                               0  ...                                            0                                   0                                      0                                     0                                     0
                   Rated 2 out of 5 stars                      0                    0                     0                       0                               0  ...                                            0                                   0                                      0                                     0                                     0
                   Rated 3 out of 5 stars                      1                    0                     1                       0                               0  ...                                            0                                   0                                      0                                     0                                     1
                   Rated 4 out of 5 stars                      0                    0                     1                       0                               0  ...                                            0                                   0                                      0                                     0                                     0
Positive Yes       Rated 3 out of 5 stars                      0                    0                     0                       0                               0  ...                                            0                                   0                                      0                                     0                                     0
                   Rated 4 out of 5 stars                      9                    8                     3                       5                               3  ...                                            0                                   0                                      0                                     0                                     1
                   Rated 5 out of 5 stars                     79                   54                    47                      32                              26  ...                                            3                                   0                                      0                                     0                                     0

[11 rows x 30 columns]

Getir

                                           cat_delivery_services  cat_product_quality  cat_customer_service  cat_pricing_and_offers  cat_online_shopping_experience  ...  cat_technical_problems_with_online_platform  cat_store_hygiene_and_organization  cat_store_environment_and_temperature  cat_policy_and_regulatory_violations  cat_unfair_product/service_practices
overall  use_again rating_raw                                                                                                                                        ...                                                                                                                                                                                                    
Negative No        Rated 1 out of 5 stars                     65                   15                    80                      20                               2  ...                                           15                                   0                                      0                                     9                                    16
                   Rated 2 out of 5 stars                      8                    4                    11                       3                               1  ...                                            3                                   0                                      0                                     0                                     1
                   Rated 3 out of 5 stars                      6                    3                     0                       1                               1  ...                                            0                                   0                                      0                                     1                                     0
                   Rated 4 out of 5 stars                      1                    0                     0                       1                               0  ...                                            0                                   0                                      0                                     0                                     0
                   Rated 5 out of 5 stars                      0                    0                     0                       1                               0  ...                                            0                                   0                                      0                                     0                                     0
         Yes       Rated 3 out of 5 stars                      1                    0                     1                       0                               0  ...                                            0                                   0                                      0                                     0                                     0
                   Rated 4 out of 5 stars                      3                    0                     1                       1                               0  ...                                            0                                   0                                      0                                     0                                     0
Positive No        Rated 5 out of 5 stars                      3                    0                     0                       0                               0  ...                                            0                                   0                                      0                                     0                                     0
         Yes       Rated 1 out of 5 stars                      2                    0                     0                       0                               0  ...                                            0                                   0                                      0                                     0                                     0
                   Rated 3 out of 5 stars                      1                    0                     0                       0                               0  ...                                            0                                   0                                      0                                     0                                     0
                   Rated 4 out of 5 stars                     16                    5                     0                       5                               1  ...                                            0                                   0                                      0                                     0                                     0
                   Rated 5 out of 5 stars                    200                   26                    33                      57                              22  ...                                            0                                   0                                      0                                     0                                     0

[12 rows x 30 columns]


Aldi

                                           cat_delivery_services  cat_product_quality  cat_customer_service  cat_pricing_and_offers  cat_online_shopping_experience  ...  cat_technical_problems_with_online_platform  cat_store_hygiene_and_organization  cat_store_environment_and_temperature  cat_policy_and_regulatory_violations  cat_unfair_product/service_practices
overall  use_again rating_raw                                                                                                                                        ...                                                                                                                                                                                                    
Negative No        Rated 1 out of 5 stars                      0                   93                   172                      21                               2  ...                                            6                                  12                                      6                                    38                                    23
                   Rated 2 out of 5 stars                      0                    6                     7                       1                               0  ...                                            1                                   3                                      0                                     1                                     2
                   Rated 3 out of 5 stars                      0                    6                     1                       1                               0  ...                                            0                                   0                                      1                                     0                                     0
                   Rated 4 out of 5 stars                      0                    0                     0                       1                               0  ...                                            0                                   0                                      0                                     0                                     0
                   Rated 5 out of 5 stars                      0                    0                     0                       0                               0  ...                                            0                                   0                                      0                                     0                                     0
         Yes       Rated 1 out of 5 stars                      0                    1                     0                       0                               0  ...                                            0                                   0                                      0                                     1                                     0
                   Rated 2 out of 5 stars                      0                    0                     0                       0                               0  ...                                            1                                   0                                      0                                     0                                     0
                   Rated 3 out of 5 stars                      0                    1                     0                       1                               0  ...                                            0                                   0                                      0                                     0                                     0
                   Rated 4 out of 5 stars                      0                    1                     0                       0                               0  ...                                            0                                   0                                      0                                     0                                     0
Positive Yes       Rated 1 out of 5 stars                      0                    0                     0                       0                               0  ...                                            0                                   0                                      0                                     0                                     0
                   Rated 4 out of 5 stars                      0                    8                     3                       3                               0  ...                                            1                                   3                                      0                                     0                                     0
                   Rated 5 out of 5 stars                      0                   25                    28                      20                               0  ...                                            0                                   8                                      1                                     0                                     0

[12 rows x 30 columns]

Sainsburys

                                           cat_delivery_services  cat_product_quality  cat_customer_service  cat_pricing_and_offers  cat_online_shopping_experience  ...  cat_technical_problems_with_online_platform  cat_store_hygiene_and_organization  cat_store_environment_and_temperature  cat_policy_and_regulatory_violations  cat_unfair_product/service_practices
overall  use_again rating_raw                                                                                                                                        ...                                                                                                                                                                                                    
Negative No        Rated 1 out of 5 stars                     25                   14                    50                       6                               2  ...                                            5                                   3                                      0                                     4                                     5
                   Rated 2 out of 5 stars                      4                    7                     4                       3                               2  ...                                            2                                   3                                      0                                     3                                     1
                   Rated 3 out of 5 stars                      5                   12                     2                       2                               0  ...                                            2                                   0                                      0                                     1                                     2
                   Rated 4 out of 5 stars                      1                    3                     0                       0                               0  ...                                            1                                   0                                      0                                     0                                     0
                   Rated 5 out of 5 stars                      1                    0                     0                       1                               0  ...                                            0                                   0                                      0                                     0                                     0
         Yes       Rated 2 out of 5 stars                      0                    0                     0                       0                               0  ...                                            0                                   0                                      0                                     0                                     0
                   Rated 3 out of 5 stars                      1                    1                     0                       0                               0  ...                                            1                                   0                                      0                                     0                                     0
                   Rated 4 out of 5 stars                      0                    0                     0                       0                               0  ...                                            0                                   0                                      0                                     0                                     0
                   Rated 5 out of 5 stars                      0                    0                     0                       0                               0  ...                                            0                                   0                                      0                                     0                                     0
Positive Yes       Rated 3 out of 5 stars                      3                    1                     0                       1                               1  ...                                            0                                   0                                      0                                     0                                     0
                   Rated 4 out of 5 stars                     20                   20                     7                       6                               8  ...                                            0                                   1                                      0                                     0                                     0
                   Rated 5 out of 5 stars                    159                   66                    58                      51                              56  ...                                            1                                   7                                      2                                     1                                     2

[12 rows x 30 columns]


"""
#=======================================================================================================================
# Graphing
#=======================================================================================================================
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

root_pickle_filepath = "/Users/jhc/personal/mba/marketing/individual_assignment/trustpilot/data1/"

filepath = "deliveroo_trustpilot_1000_reviews_raw_data_exploded_categories_4.pickle"
company_name = "deliveroo"

filepath = "ocado_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
company_name = "ocado"

filepath = "tesco_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
company_name = "tesco"

filepath = "aldi_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
company_name = "aldi"

filepath = "sainsburys_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
company_name = "sainsburys"

filepath = "getir_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
company_name = "getir"

filepath = "morrisons_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
company_name = "morrisons"

filepath = "waitrose_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
company_name = "waitrose"

filepath = "gousto_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
company_name = "gousto"

filepath = "just_eat_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
company_name = "just_eat"

# 5 most diverse:
# Ocado -
# Aldi - Discount
# Gousto - Meal kit
# Deliveroo - Takeaway
# Sainsburys - Traditional supermarket
# Getir - Quick Commerce



df = pd.read_pickle(filepath)
category_columns = [col for col in df.columns if col.startswith('cat_')]
gcols = ["overall"] + category_columns
df = df[gcols]
# Reverse the underscores in the columns and remove the cat_
df.columns = df.columns.str.replace('cat_', '').str.replace('_', ' ').str.title()
# rename the Overall column overall
df.rename(columns={"Overall": "overall"}, inplace=True)
category_columns = [col for col in df.columns if col != 'overall']
grouped = df.groupby('overall')[category_columns].sum().reset_index()

negative_counts = grouped[grouped['overall'] == 'Negative'][category_columns].sum()
positive_counts = grouped[grouped['overall'] == 'Positive'][category_columns].sum()


# Create a function to replace 0 with an empty string
def replace_zero_with_empty_string(counts):
    return [str(count) if count != 0 else '' for count in counts]

# Create the bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    y=category_columns,
    x=negative_counts,
    name='Negative',
    orientation='h',
    marker=dict(color='red'),
    text=replace_zero_with_empty_string(negative_counts),
    textposition='outside'
))

fig.add_trace(go.Bar(
    y=category_columns,
    x=positive_counts,
    name='Positive',
    orientation='h',
    marker=dict(color='green'),
    text=replace_zero_with_empty_string(positive_counts),
    textposition='outside'
))

# Update layout
fig.update_layout(
    title=dict(
        text=f'{company_name.title()}: Negative and Positive Counts for Each Review Category',
        x=0.5
    ),
    yaxis_title='Category',
    xaxis_title='Number of Reviews',
    barmode='group',
    # paper_bgcolor='white',
    # plot_bgcolor='white',
    # Make background transparent:
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showgrid=True, gridcolor='purple'),
    width=11.69 * 150,  # converting inches to pixels (assuming 96 dpi)
    height=8.27 * 150  # converting inches to pixels (assuming 96 dpi)
)

#filename = f'{company_name}_positive_negative.png'
filename = f'{company_name}_positive_negative_transparent.png'
print(filename)
pio.write_image(fig, filename, format='png')

# Show the figure
fig.show()

# /Users/jhc/personal/mba/marketing/individual_assignment/trustpilot/data1


# image_bytes = pio.to_image(fig, format='png', engine='kaleido', width=800, height=600, scale=2, transparent=True)
#
# with open('my_plot.png', 'wb') as f:
#     f.write(image_bytes)
#=======================================================================================================================


# Hygiene factors and motivators



# Motivators
motivators = [
    'Product Quality',
    'Customer Service',
    'Online Shopping Experience',
    'Loyalty Programs',
    'Value For Money',
    'Product Range Diversity',
    'Pricing And Offers',
]

# Hygiene Factors
hygiene_factors = [
    'Store Experience And Layout',
    'Delivery Services',
    'Click & Collect Service',
    'Website And App Interface',
    'Packaging And Handling',
    'Stock Availability',
    'Communication/Information Transparency',
    'Employee/Driver Behavior',
    'Overcharging/Double Charging',
    'Substitution Policy',
    'Refund/Return Process',
    'Food Hygiene And Safety',
    'Delivery Cancellations/Missing Items',
    'Product Expiration Dates',
    'Payment And Billing Issues',
    'Parking And Disability Access',
    'Complaint Handling',
    'Misleading Pricing And Advertisement',
    'Technical Problems With Online Platform',
    'Store Hygiene And Organization',
    'Store Environment And Temperature',
    'Policy And Regulatory Violations',
    'Unfair Product/Service Practices'
]
print(f"motivators: {len(motivators)}")
print(f"hygiene_factors: {len(hygiene_factors)}")

def make_dirs(dirname):
    """Make a directory and any necessary parent directory.  If directory
    already exists, do nothing"""
    try:
        # Make sure that the dest_dir exists (create it if not)
        os.makedirs(dirname)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

graph_base_filepath = "/Users/jhc/personal/mba/marketing/individual_assignment/trustpilot/data1/graph_tests"

# Define size categories
small_threshold = 50
medium_threshold = 150

import errno

# Function to determine size category
def get_size_category(value):
    if value <= small_threshold:
        return 'small'
    elif value <= medium_threshold:
        return 'medium'
    else:
        return 'large'


# Determine size category for each column
size_categories = column_sums.apply(get_size_category)

# Define sizes for the categories
size_mapping = {
    'small': 300,
    'medium': 400,
    'large': 500
}

ocado_filepath = "ocado_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
ocado_company_name = "ocado"

sainsburys_filepath = "sainsburys_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
sainsburys_company_name = "sainsburys"

gousto_filepath = "gousto_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
gousto_company_name = "gousto"

deliveroo_filepath = "deliveroo_trustpilot_1000_reviews_raw_data_exploded_categories_4.pickle"
deliveroo_company_name = "deliveroo"

aldi_filepath = "aldi_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
aldi_company_name = "aldi"

getir_filepath = "getir_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
getir_company_name = "getir"



all_filepath_company_names = [
    (ocado_filepath, ocado_company_name),
    (sainsburys_filepath, sainsburys_company_name),
    (gousto_filepath, gousto_company_name),
    (deliveroo_filepath, deliveroo_company_name),
    (aldi_filepath, aldi_company_name),
    (getir_filepath, getir_company_name),
]

root_pickle_filepath = "/Users/jhc/personal/mba/marketing/individual_assignment/trustpilot/data1/"

#=======================================================================================================================
# Motivators and Hygiene Factors Graphing with different size pie charts
#=======================================================================================================================


for filepath, company_name in all_filepath_company_names:
    pickle_filepath = os.path.join(root_pickle_filepath, filepath)
    df = pd.read_pickle(pickle_filepath)
    category_columns = [col for col in df.columns if col.startswith('cat_')]
    gcols = ["overall"] + category_columns
    df = df[gcols]
    # Reverse the underscores in the columns and remove the cat_
    df.columns = df.columns.str.replace('cat_', '').str.replace('_', ' ').str.title()
    # rename the Overall column overall
    df.rename(columns={"Overall": "overall"}, inplace=True)

    df = df.groupby("overall").sum()
    df = df[motivators]


    company_base_filepath= os.path.join(graph_base_filepath, company_name)
    make_dirs(company_base_filepath)

    column_sums = df.sum()

    for column in df.columns:
        size_category = size_categories[column]
        size_category = 'small'
        size = size_mapping[size_category]

        #text = [f'{val} ({val / sum(df[column]) * 100:.1f}%)' if val != 0 else '' for val in df[column]]
        text = [f'{val}' if val != 0 else '' for val in df[column]]

        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=df.index,
            values=df[column],
            #textinfo='value+percent',
            #textinfo='value',
            text=text,
            textinfo='text',
            #title=column,
            #hole=0.4
            #marker=dict(colors=['darkpurple', 'darkgreen']),
            marker=dict(colors=['darkred', 'darkgreen']),
            pull=[0, 0.1]  # Pull the positive slice away
        ))

        # Adjust the size of each figure based on the size category
        fig.update_layout(
            # Make background transparent:
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            width=size,
            height=size,
            showlegend=False,
            title=dict(
                text=f'{column}',
                #text=f'{column}<br>Reviews: {column_sums[column]}',
                x=0.5,
                #y=0.8,
            ),
            # annotations=[
            #     dict(
            #         text=f'Sum: {column_sums[column]}',
            #         x=0.5,
            #         y=-0.1,
            #         showarrow=False
            #     )
            # ]
        )

        # Show the figure
        #fig.show()
        filename = f"{column.replace(' ', '_')}_pie_chart.png"
        filepath = os.path.join(company_base_filepath, filename)
        print(filepath)
        fig.write_image(filepath)

size = 300
df = pd.DataFrame({
    'Category': ['Negative', 'Positive'],
    'Product Quality': [20, 81]
})
df.set_index('Category', inplace=True)
column='Product Quality'

size = 300
df = pd.DataFrame({
    'Category': ['Negative', 'Positive'],
    'Product Range Diversity': [5, 60]
})
df.set_index('Category', inplace=True)
column='Product Range Diversity'



text = [f'{val}' if val != 0 else '' for val in df[column]]

filename = f"product_quality_pie_chart.png"
filename = f"product_range_diversity_pie_chart.png"
#filepath = os.path.join(company_base_filepath, filename)
filepath = filename
print(filepath)
fig.write_image(filepath)

#=======================================================================================================================
# Hygiene Factors Graphing bar charts
#=======================================================================================================================

root_pickle_filepath = "/Users/jhc/personal/mba/marketing/individual_assignment/trustpilot/data1/"

ocado_filepath = "ocado_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
ocado_company_name = "ocado"

sainsburys_filepath = "sainsburys_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
sainsburys_company_name = "sainsburys"

gousto_filepath = "gousto_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
gousto_company_name = "gousto"

deliveroo_filepath = "deliveroo_trustpilot_1000_reviews_raw_data_exploded_categories_4.pickle"
deliveroo_company_name = "deliveroo"

aldi_filepath = "aldi_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
aldi_company_name = "aldi"

getir_filepath = "getir_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
getir_company_name = "getir"

tesco_filepath = "tesco_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
tesco_company_name = "tesco"

morrisons_filepath = "morrisons_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
morrisons_company_name = "morrisons"

waitrose_filepath = "waitrose_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
waitrose_company_name = "waitrose"

just_eat_filepath = "just_eat_trustpilot_550_reviews_raw_data_exploded_categories_4.pickle"
just_eat_company_name = "just_eat"

all_filepath_company_names = [
    (ocado_filepath, ocado_company_name),
    (sainsburys_filepath, sainsburys_company_name),
    (gousto_filepath, gousto_company_name),
    (deliveroo_filepath, deliveroo_company_name),
    (aldi_filepath, aldi_company_name),
    (getir_filepath, getir_company_name),
    (tesco_filepath, tesco_company_name),
    (morrisons_filepath, morrisons_company_name),
    (waitrose_filepath, waitrose_company_name),
    (just_eat_filepath, just_eat_company_name)
]

# Read all pickle files into dataframes
all_dfs = [pd.read_pickle(os.path.join(root_pickle_filepath, filepath)) for filepath, company_name in all_filepath_company_names]

# Concatenate the dataframes
df = pd.concat([df for df in all_dfs])

category_columns = [col for col in df.columns if col.startswith('cat_')]
category_columns2 = [col.replace('cat_', '').replace('_', ' ').title() for col in category_columns]
hygiene_factor_category_columns2 = [col for col in category_columns2 if col in hygiene_factors]
df.columns = df.columns.str.replace('cat_', '').str.replace('_', ' ').str.title()

df.rename(columns={"Overall": "overall"}, inplace=True)
grouped = df.groupby('overall')[hygiene_factor_category_columns2].sum().reset_index()
negative_counts = grouped[grouped['overall'] == 'Negative'][hygiene_factor_category_columns2].sum()
positive_counts = grouped[grouped['overall'] == 'Positive'][hygiene_factor_category_columns2].sum()
negative_name = "Negative"
positive_name = "Positive"
title_text = "All Companies (5,000 Reviews): Negative and Positive Counts for Each Review Category"


df.rename(columns={"Use Again": "use_again"}, inplace=True)
grouped = df.groupby('use_again')[hygiene_factor_category_columns2].sum().reset_index()
negative_counts = grouped[grouped['use_again'] == 'No'][hygiene_factor_category_columns2].sum()
positive_counts = grouped[grouped['use_again'] == 'Yes'][hygiene_factor_category_columns2].sum()
negative_name = "No"
positive_name = "Yes"
title_text = "All Companies (5,000 Reviews): Would you use this service again? Yes/No Counts for Each Review Category"




# Create a function to replace 0 with an empty string
def replace_zero_with_empty_string(counts):
    return [str(count) if count != 0 else '' for count in counts]


import plotly.graph_objs as go
import plotly.io as pio



# Create the bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    y=hygiene_factor_category_columns2,
    x=negative_counts,
    name=negative_name,
    orientation='h',
    marker=dict(color='darkred'),
    text=replace_zero_with_empty_string(negative_counts),
    textposition='outside'
))

fig.add_trace(go.Bar(
    y=hygiene_factor_category_columns2,
    x=positive_counts,
    name=positive_name,
    orientation='h',
    marker=dict(color='darkgreen'),
    text=replace_zero_with_empty_string(positive_counts),
    textposition='outside'
))

# Update layout
fig.update_layout(
    title=dict(
        text=title_text,
        x=0.5
    ),
    yaxis_title='Category',
    xaxis_title='Number of Reviews',
    barmode='group',
    # paper_bgcolor='white',
    # plot_bgcolor='white',
    # Make background transparent:
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showgrid=True, gridcolor='purple'),
    #width=11.69 * 150,  # converting inches to pixels (assuming 96 dpi)
    #height=8.27 * 150  # converting inches to pixels (assuming 96 dpi)
    height = 11.69 * 150,  # converting inches to pixels (assuming 96 dpi)
    width = 8.27 * 170,  # converting inches to pixels (assuming 96 dpi)
    # Make text bigger
    font=dict(size=18),
    #showlegend=False,
    showlegend=True,
)

fig.show()

company_name = "all_companies"
#filename = f'{company_name}_positive_negative.png'
filename = f'{company_name}_{positive_name}_{negative_name}_transparent.png'
print(filename)
pio.write_image(fig, filename, format='png')

# Show the figure
#=======================================================================================================================

"""
Data from:
https://0-www-statista-com.pugwash.lib.warwick.ac.uk/statistics/1341162/most-well-known-online-grocery-delivery-brands-in-the-uk/
"""

import plotly.graph_objects as go
import plotly.io as pio

data = {
    "Tesco": 96,
    "Asda": 96,
    "Morrisons": 94,
    "Sainsbury's": 94,
    "Co-op": 93,
    "Waitrose": 90,
    "HelloFresh": 80,
    "Ocado": 75,
    "Gousto": 61,
    "Getir": 31,
    "Gorillas": 28,
    "Zapp": 22
}

trad_blue = "#2690D4"
discount_dark_blue = "#004AAD"
ocado_purple = "#56216E"
quick_commerce_orange = "#EEB033"
takeaway_red = "#E7191F"
meal_kit_green = "#9CC34A"

data_colours = {
    "Tesco": trad_blue,
    "Asda": trad_blue,
    "Morrisons": trad_blue,
    "Sainsbury's": trad_blue,
    "Co-op": trad_blue,
    "Waitrose": trad_blue,
    "HelloFresh": meal_kit_green,
    "Ocado": ocado_purple,
    "Gousto": meal_kit_green,
    "Getir": quick_commerce_orange,
    "Gorillas": quick_commerce_orange,
    "Zapp": quick_commerce_orange,
}

fig = go.Figure()

for store, value in data.items():
    fig.add_trace(go.Bar(
        y=[store],
        x=[value],
        orientation='h',
        marker=dict(color=data_colours[store]),
        name=store,
        text=f'{value}%',
        textposition='outside',
    ))

title_text = "Leading online grocery delivery brands ranked by brand awareness in the United Kingdom in 2023"

fig.update_layout(
    title=dict(
        text=title_text,
        x=0.5
    ),
    yaxis_title='Company',
    yaxis=dict(autorange='reversed'),
    xaxis_title='Share of respondents',
    barmode='stack',
    # Make background transparent:
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showgrid=True, gridcolor='purple'),
    height = 11.69 * 150,  # converting inches to pixels (assuming 96 dpi)
    width = 8.27 * 170,  # converting inches to pixels (assuming 96 dpi)
    # Make text bigger
    font=dict(size=18),
    showlegend=False,
    bargap=0.7,
)

#fig.show()
filename = f'brand_awareness_total_transparent.png'
print(filename)
pio.write_image(fig, filename, format='png')


#=======================================================================================================================
# Brand Awareness Graphing Difference
#=======================================================================================================================

# Ocado
# https://0-www-statista-com.pugwash.lib.warwick.ac.uk/forecasts/1304373/ocado-online-grocery-delivery-brand-profile-in-the-united-kingdom

import plotly.graph_objects as go
import pandas as pd

# Define the data
tesco_data = {
    "Tesco awareness": 97,
    "Tesco popularity": 60,
    "Tesco usage": 51,
    "Tesco loyalty": 45,
    "Tesco buzz": 34
}
ocado_data = {
    "Ocado awareness": 75,
    "Ocado popularity": 16,
    "Ocado usage": 13,
    "Ocado loyalty": 11,
    "Ocado buzz": 17
}
ocado_data = {
    "Ocado awareness": 85,
    "Ocado popularity": 32,
    "Ocado usage": 26,
    "Ocado loyalty": 22,
    "Ocado buzz": 34
}
getir_data = {
    "Getir awareness": 32,
    "Getir popularity": 6,
    "Getir usage": 5,
    "Getir loyalty": 3,
    "Getir buzz": 6
}
aldi_data = {
    "Aldi awareness": 95,
    "Aldi popularity": 52,
    "Aldi usage": 53,
    "Aldi loyalty": 48,
    "Aldi buzz": 38
}
gousto_data = {
    "Gousto awareness": 64,
    "Gousto popularity": 10,
    "Gousto usage": 8,
    "Gousto loyalty": 5,
    "Gousto buzz": 18
}

# Create a DataFrame with the correct index order
df_corrected_ordered = pd.DataFrame({
    "awareness": [tesco_data["Tesco awareness"], aldi_data["Aldi awareness"], ocado_data["Ocado awareness"], gousto_data["Gousto awareness"], getir_data["Getir awareness"]],
    "popularity": [tesco_data["Tesco popularity"], aldi_data["Aldi popularity"], ocado_data["Ocado popularity"], gousto_data["Gousto popularity"], getir_data["Getir popularity"]],
    "usage": [tesco_data["Tesco usage"], aldi_data["Aldi usage"], ocado_data["Ocado usage"], gousto_data["Gousto usage"], getir_data["Getir usage"]],
    "loyalty": [tesco_data["Tesco loyalty"], aldi_data["Aldi loyalty"], ocado_data["Ocado loyalty"], gousto_data["Gousto loyalty"], getir_data["Getir loyalty"]],
    "buzz": [tesco_data["Tesco buzz"], aldi_data["Aldi buzz"], ocado_data["Ocado buzz"], gousto_data["Gousto buzz"], getir_data["Getir buzz"]]
}, index=["Tesco", "Aldi", "Ocado", "Gousto", "Getir"])

df_corrected_ordered.columns = df_corrected_ordered.columns.str.title()

# Colors for each brand
trad_blue = "#2690D4"  # tesco
discount_dark_blue = "#004AAD"  # aldi
ocado_purple = "#56216E"  # ocado
quick_commerce_orange = "#EEB033"  # getir
meal_kit_green = "#9CC34A"  # Gousto

# Create the figure
fig = go.Figure()

# Add bar traces for each brand
fig.add_trace(go.Bar(
    name='Tesco',
    x=df_corrected_ordered.columns,
    y=df_corrected_ordered.loc['Tesco'],
    marker_color=trad_blue
))

fig.add_trace(go.Bar(
    name='Aldi',
    x=df_corrected_ordered.columns,
    y=df_corrected_ordered.loc['Aldi'],
    marker_color=discount_dark_blue
))

fig.add_trace(go.Bar(
    name='Ocado',
    x=df_corrected_ordered.columns,
    y=df_corrected_ordered.loc['Ocado'],
    marker_color=ocado_purple
))

fig.add_trace(go.Bar(
    name='Gousto',
    x=df_corrected_ordered.columns,
    y=df_corrected_ordered.loc['Gousto'],
    marker_color=meal_kit_green
))

fig.add_trace(go.Bar(
    name='Getir',
    x=df_corrected_ordered.columns,
    y=df_corrected_ordered.loc['Getir'],
    marker_color=quick_commerce_orange
))

title_text = "Brand awareness, usage, popularity, loyalty, and buzz among online grocery delivery users in the United Kingdom in 2022"
title_text = "Brand awareness Metrics 6 months after Marketing Plan Start"

fig.update_layout(
    title=dict(
        text=title_text,
        x=0.5
    ),
    yaxis_title='Share of respondents',
    #yaxis=dict(autorange='reversed'),
    xaxis_title='Brand Metrics',
    barmode='group',
    # Make background transparent:
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    yaxis=dict(showgrid=True, gridcolor='purple'),
    width = 11.69 * 150,  # converting inches to pixels (assuming 96 dpi)
    height = 8.27 * 170,  # converting inches to pixels (assuming 96 dpi)
    # Make text bigger
    font=dict(size=18),
    showlegend=False,
    #showlegend=True,
    bargap=0.5,
)

fig.show()

#filename = f'brand_awareness_stages_transparent.png'
filename = f'brand_awareness_stages_afterwards_transparent.png'
print(filename)
pio.write_image(fig, filename, format='png')

#=======================================================================================================================
#
#=======================================================================================================================

















# API Error string:
# Error: You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.






reviews_df['Detailed Category'] = reviews_df['body_text'].apply(lambda x: categorize_review(x, categories))

# Apply the function to categorize each review
reviews_df['Detailed Category'] = reviews_df['body_text'].apply(lambda x: categorize_review(x, categories))

# Save the categorized reviews to a new Excel file
output_file_path = 'categorized_reviews2.xlsx'
reviews_df.to_excel(output_file_path, index=False)

print(f"Categorized reviews have been saved to {output_file_path}")


def get_experience_date(x):
    try:
        return parser.parse(x.split(":")[1].strip())
    except:
        return None

df['review_datetime'] = pd.to_datetime(df['review_datetime_raw'])
df['review_datetime'] = df['review_datetime'].dt.tz_localize(None)
df["rating_score"] = df["rating_raw"].apply(lambda x: int(x.split(" ")[1].strip()))
df["experience_datetime"] = df["experience_date_raw"].apply(get_experience_date)

df = df[["review_datetime_raw", "experience_date_raw", "rating_raw", "review_datetime", "experience_datetime", "rating_score", "review_title", "review_body"]]
df_raw2 = df.copy()
df2 = df.copy()
filepath_pickle2 = "{}/{}_trustpilot_{}_reviews_{}_2.pkl".format(base_filepath, company_name, num_reviews, "raw_data")
df.to_pickle(filepath_pickle2)
print(filepath_pickle2)
#filepath_pickle = "/Users/jhc/personal/mba/operations_management/group_assignment/data/data_with_datetime/booking_com_trustpilot_1000_reviews_raw_data_2.pkl"
#df = pd.read_pickle(filepath_pickle)

import pandas as pd
filepath_pickle_booking = "/Users/jhc/personal/mba/operations_management/group_assignment/data/data_with_datetime/booking_com_trustpilot_1000_reviews_raw_data_2.pkl"
dfb = pd.read_pickle(filepath_pickle_booking)
filepath_pickle_expedia = "/Users/jhc/personal/mba/operations_management/group_assignment/data/data_with_datetime/expedia_com_trustpilot_1000_reviews_raw_data_2.pkl"
dfe = pd.read_pickle(filepath_pickle_expedia)

df2 = dfe
df2 = dfb

all_rating_counts_df = df2.rating_score.value_counts().to_frame()
all_rating_counts_df.reset_index(inplace=True)


# Control Limit Constants
# From: https://towardsdatascience.com/quality-control-charts-guide-for-python-9bb1c859c051
data = {
    "subgroup_size_n": range(2, 26),
    "A2": [1.880, 1.023, 0.729, 0.577, 0.483, 0.419, 0.373, 0.337, 0.308, 0.285, 0.266, 0.249, 0.235, 0.223, 0.212, 0.203, 0.194, 0.187, 0.180, 0.173, 0.167, 0.162, 0.157, 0.153],
    "A3": [2.659, 1.954, 1.628, 1.427, 1.287, 1.182, 1.099, 1.032, 0.975, 0.927, 0.886, 0.850, 0.817, 0.789, 0.763, 0.739, 0.718, 0.698, 0.680, 0.663, 0.647, 0.633, 0.619, 0.606],
    "d2": [1.128, 1.693, 2.059, 2.326, 2.534, 2.704, 2.847, 2.970, 3.078, 3.173, 3.258, 3.336, 3.407, 3.472, 3.532, 3.588, 3.640, 3.689, 3.735, 3.778, 3.819, 3.858, 3.895, 3.931],
    "D3": [0.000, 0.000, 0.000, 0.000, 0.000, 0.076, 0.136, 0.184, 0.223, 0.256, 0.283, 0.307, 0.328, 0.347, 0.363, 0.378, 0.391, 0.403, 0.415, 0.425, 0.434, 0.443, 0.451, 0.459],
    "D4": [3.267, 2.574, 2.282, 2.114, 2.004, 1.924, 1.864, 1.816, 1.777, 1.744, 1.717, 1.693, 1.672, 1.653, 1.637, 1.622, 1.608, 1.597, 1.585, 1.575, 1.566, 1.557, 1.548, 1.541],
    "B3": [0.000, 0.000, 0.000, 0.000, 0.030, 0.118, 0.185, 0.239, 0.284, 0.321, 0.354, 0.382, 0.406, 0.428, 0.448, 0.466, 0.482, 0.497, 0.510, 0.523, 0.534, 0.545, 0.555, 0.565],
    "B4": [3.267, 2.568, 2.266, 2.089, 1.970, 1.882, 1.815, 1.761, 1.716, 1.679, 1.646, 1.618, 1.594, 1.572, 1.552, 1.534, 1.518, 1.503, 1.490, 1.477, 1.466, 1.455, 1.445, 1.435]
}

df_stats = pd.DataFrame(data)


df = dfe
df = dfb
"""
Learning SPC calculations: https://youtu.be/Aj7lJLR-7b4?si=EuoAf1QdJokyR5Yp
"""

def get_freq_df(freq):
    weekly_rating = df.groupby([pd.Grouper(key='review_datetime', freq=freq)]).agg(
        rating_scores=('rating_score', list),
        total_reviews=('rating_score', 'count')
    )
    def random_sample(values):
        # Function to randomly select up to 25 sample values from the list
        return random.sample(values, min(len(values), 25))

    # Adding a column with random samples
    weekly_rating['sample_ratings'] = weekly_rating['rating_scores'].apply(random_sample)

    # Adding a column for the count of values in the sample
    weekly_rating['sample_count'] = weekly_rating['sample_ratings'].apply(len)

    weekly_rating.reset_index(inplace=True)

    weekly_rating['sample_ratings'] = weekly_rating['sample_ratings'].apply(lambda x: sorted(x))
    weekly_rating['sample_x_bar'] = weekly_rating['sample_ratings'].apply(lambda x: np.mean(x))
    #weekly_rating['sample_range_r'] = weekly_rating['sample_ratings'].apply(lambda x: max(x) - min(x))
    def get_range(x):
        try:
            return max(x) - min(x)
        except ValueError:
            return 0
    weekly_rating['sample_range_r'] = weekly_rating['sample_ratings'].apply(get_range)

    # center line:
    cl = np.mean(weekly_rating['sample_x_bar'])
    r_bar = np.mean(weekly_rating['sample_range_r'])
    weekly_rating['center_line'] = cl
    weekly_rating['r_bar'] = r_bar

    a2_dict = pd.Series(df_stats.A2.values, index=df_stats.subgroup_size_n).to_dict()
    weekly_rating['sample_A2_value'] = weekly_rating['sample_count'].map(a2_dict)

    weekly_rating["x_bar_chart_sample_ucl"] = cl + (weekly_rating['sample_A2_value'] * r_bar)
    weekly_rating["x_bar_chart_sample_lcl"] = cl - (weekly_rating['sample_A2_value'] * r_bar)

    d3_dict = pd.Series(df_stats.D3.values, index=df_stats.subgroup_size_n).to_dict()
    d4_dict = pd.Series(df_stats.D4.values, index=df_stats.subgroup_size_n).to_dict()

    weekly_rating['sample_D3_value'] = weekly_rating['sample_count'].map(d3_dict)
    weekly_rating['sample_D4_value'] = weekly_rating['sample_count'].map(d4_dict)

    weekly_rating["r_chart_sample_ucl"] = cl + (weekly_rating['sample_D4_value'] * r_bar)
    weekly_rating["r_chart_sample_lcl"] = cl - (weekly_rating['sample_D3_value'] * r_bar)
    return weekly_rating

weekly_3_rating_counts_df = get_freq_df('3W')
weekly_2_rating_counts_df = get_freq_df('2W')
weekly_rating_counts_df = get_freq_df('W')
daily_3_rating_counts_df = get_freq_df('3D')
daily_rating_counts_df = get_freq_df('D')


#=======================================================================================================================
# Output to Excel
#=======================================================================================================================

filepath_spreadsheet = f"{base_filepath}/{company_name}_trustpilot_{num_reviews}_reviews_processed.xlsx"
print(filepath_spreadsheet)

with pd.ExcelWriter(filepath_spreadsheet, engine='openpyxl') as writer:
    df2.to_excel(writer, index=False, sheet_name='All {}'.format(company_name))
    for i in range(1, 6):
        df2[df2["rating_score"] == i].to_excel(writer, index=False, sheet_name='Rating {}'.format(i))

    weekly_3_rating_counts_df.to_excel(writer, index=False, sheet_name='3 Week Rating Counts')
    weekly_2_rating_counts_df.to_excel(writer, index=False, sheet_name='2 Week Rating Counts')
    weekly_rating_counts_df.to_excel(writer, index=False, sheet_name='Weekly Rating Counts')
    daily_3_rating_counts_df.to_excel(writer, index=False, sheet_name='3 Day Window Rating Counts')
    daily_rating_counts_df.to_excel(writer, index=False, sheet_name='Daily Rating Counts')
    all_rating_counts_df.to_excel(writer, index=False, sheet_name='All Rating Counts')

print(filepath_spreadsheet)


#=======================================================================================================================
# Graphing
#=======================================================================================================================

def draw_graph(dfg, period, title, company_name_title):
    dfg = dfg[['review_datetime', 'center_line', 'sample_x_bar', 'x_bar_chart_sample_ucl', 'x_bar_chart_sample_lcl']]
    dfg['review_datetime'] = pd.to_datetime(dfg['review_datetime'])
    dfg = dfg.sort_values('review_datetime')

    fig = go.Figure()

    center_line_colour = 'navy'
    sample_x_bar_colour = 'blue'
    #center_line_colour = 'rgb(191,155,48)'
    #sample_x_bar_colour = 'rgb(255,191,0)'

    fig.add_trace(go.Scatter(x=dfg['review_datetime'], y=dfg['center_line'], mode='lines', name='Center Line',
                             line=dict(color=center_line_colour, dash='solid')))

    fig.add_trace(go.Scatter(x=dfg['review_datetime'], y=dfg['sample_x_bar'], mode='lines+markers',
                             name='Review Rating (Sample X Bar)', line=dict(color=sample_x_bar_colour, dash='dashdot')))
    fig.add_trace(
        go.Scatter(x=dfg['review_datetime'], y=dfg['x_bar_chart_sample_ucl'], mode='lines+markers', name='Sample UCL',
                   line=dict(color='green', dash='dot')))
    fig.add_trace(
        go.Scatter(x=dfg['review_datetime'], y=dfg['x_bar_chart_sample_lcl'], mode='lines+markers', name='Sample LCL',
                   line=dict(color='green', dash='dot')))

    fig.update_layout(
        title={
            'text': f'{company_name_title} Statistical Process Control (SPC) Review Rating {title} X-bar Chart',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title=f'Date Review Made ({title})',
        yaxis_title=f'X-bar of Review Sample',
        plot_bgcolor='rgb(255, 255, 255)',  # White
        paper_bgcolor='rgb(255, 255, 255)',  # White background outside the plot
        font=dict(
            family="Arial, sans-serif",
            size=16,
            color="black"
        ),
        xaxis=dict(
            showline=True,
            showgrid=False,
            linecolor='navy'
        ),
        yaxis=dict(
            showline=True,
            showgrid=True,
            gridcolor='teal',
            linecolor='navy',
            range=[0, 2]
        )
    )

    fig.show()
    output_filename = f"{base_filepath}/{company_name}_trustpilot_{num_reviews}_reviews_{period}_spc_x_bar_chart_2.png"
    width = 1920
    height = 1080
    pio.write_image(fig, output_filename, width=width, height=height, scale=2)
    print(output_filename)


# period => dataframe
datasets = [
    {'period': 'weekly_3', 'dff': weekly_3_rating_counts_df, 'title': '3 Week Window'},
    {'period': 'weekly_2', 'dff': weekly_2_rating_counts_df, 'title': '2 Week Window'},
    {'period': 'weekly', 'dff': weekly_rating_counts_df, 'title': 'Weekly Window'},
    {'period': 'daily_3', 'dff': daily_3_rating_counts_df, 'title': '3 Day Window'},
    {'period': 'daily', 'dff': daily_rating_counts_df, 'title': 'Daily Window'},
]

#datasets = [datasets[-1]]


datasets = [
    #{'period': 'weekly_2', 'dff': weekly_2_rating_counts_df, 'title': '2 Week Window'},
    {'period': 'weekly_3', 'dff': weekly_3_rating_counts_df, 'title': '3 Week Window'},
]

datasets = [
{'period': 'daily_3', 'dff': daily_3_rating_counts_df, 'title': '3 Day Window'},
]

for dataset in datasets:
    period = dataset['period']
    dff = dataset['dff']
    title = dataset['title']
    draw_graph(dff, period, title, company_name_title)


#=======================================================================================================================
# Graphing introductory data
#=======================================================================================================================

company_name_title = "Expedia.com"
filepath_pickle = "/Users/jhc/personal/mba/operations_management/group_assignment/data/data_with_datetime/expedia_com_trustpilot_1000_reviews_raw_data_2.pkl"
freq = '3W'
period = "weekly_3"
period_text = "3-Week Groups"
# https://www.color-hex.com/color-palette/2799
colours = ['#a67c00', '#bf9b30', '#ffbf00', '#ffcf40', '#ffdc73']

company_name_title = "Booking.com"
filepath_pickle = "/Users/jhc/personal/mba/operations_management/group_assignment/data/data_with_datetime/booking_com_trustpilot_1000_reviews_raw_data_2.pkl"
freq = '3D'
period = "daily_3"
period_text = "3-Day Groups"
colours = ['navy', '#7499ff', '#7fa8ff', '#a4c1ff', '#c4d7ff']


df = pd.read_pickle(filepath_pickle)

df.set_index('review_datetime', inplace=True)
df_resampled = df.groupby([pd.Grouper(freq=freq), 'rating_score']).size().unstack(fill_value=0)

traces = []
for i, rating in enumerate(sorted(df_resampled.columns, reverse=True)):
    traces.append(go.Bar(
        x=df_resampled.index.strftime('%Y-%m-%d'),  # Format for readability
        y=df_resampled[rating],
        name=f'Rating {rating}',
        marker_color=colours[i]
    ))

# Create the figure and add traces
fig = go.Figure(data=traces)

# Stack the bars
fig.update_layout(barmode='stack')

xaxis_title = f'Date Review Made ({period_text})'

time_period = (df.index.max() - df.index.min()).days
num_reviews = len(df)

# Customize the layout
fig.update_layout(
    title={
        'text': f'{company_name_title} Review Ratings ({num_reviews} reviews over {time_period} days)',
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title=xaxis_title,
    yaxis_title='Rating Count',
    #xaxis={'type': 'category'},  # Treat x-axis values as discrete categories
    paper_bgcolor='rgb(255, 255, 255)',  # White
    plot_bgcolor='rgb(255, 255, 255)',  # White
    font=dict(
        family="Arial, sans-serif",
        size=16,
        color="black"
    ),
    xaxis=dict(
        showline=True,
        showgrid=False,
        linecolor='navy',
        type='category',
        tickangle=45,
        tickmode='auto',
        #nticks=20,
    ),
    yaxis=dict(
        showline=True,
        showgrid=True,
        gridcolor='teal',
        linecolor='navy'
    ),
    width=1000,
    bargap=0.4,
    showlegend=True,
)

# Show the figure
fig.show()


#=======================================================================================================================
# Graphing introductory data - Percentages:
#=======================================================================================================================
df_percentage = df_resampled.div(df_resampled.sum(axis=1), axis=0) * 100
df_percentage['negative'] = df_percentage[1]
df_percentage['neutral'] = df_percentage[2] + df_percentage[3]
df_percentage['positive'] = df_percentage[4] + df_percentage[5]
dfg = df_percentage[['negative', 'neutral', 'positive']]
dfg = df_percentage[['neutral', 'positive']]
dfg = df_percentage[['positive', 'neutral']]

columns = dfg.columns

columns = sorted(dfg.columns, reverse=True)
columns = ['positive', 'neutral']
dfg = df_percentage[columns]


traces = []
for i, rating in enumerate(columns):
    print(i, rating)
    traces.append(go.Bar(
        x=dfg.index.strftime('%Y-%m-%d'),  # Format for readability
        y=dfg[rating],
        name=f'Rating {rating}',
        marker_color=colours[i]
    ))


# Create the figure and add traces
fig = go.Figure(data=traces)

# Stack the bars
fig.update_layout(barmode='stack')

xaxis_title = f'Date Review Made ({period_text})'

time_period = (dfg.index.max() - dfg.index.min()).days
num_reviews = len(df)

# Customize the layout
fig.update_layout(
    title={
        'text': f'{company_name_title} Review Ratings ({num_reviews} reviews over {time_period} days)',
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title=xaxis_title,
    yaxis_title='Rating Count',
    #xaxis={'type': 'category'},  # Treat x-axis values as discrete categories
    paper_bgcolor='rgb(255, 255, 255)',  # White
    plot_bgcolor='rgb(255, 255, 255)',  # White
    font=dict(
        family="Arial, sans-serif",
        size=16,
        color="black"
    ),
    xaxis=dict(
        showline=True,
        showgrid=False,
        linecolor='navy',
        type='category',
        tickangle=45,
        tickmode='auto',
        #nticks=20,
    ),
    yaxis=dict(
        showline=True,
        showgrid=True,
        gridcolor='teal',
        linecolor='navy'
    ),
    width=1000,
    bargap=0.4,
    showlegend=True,
)

# Show the figure
fig.show()



#=======================================================================================================================
# Graphing introductory data - Percentages:
#=======================================================================================================================
from plotly.subplots import make_subplots

df_percentage = df_resampled.div(df_resampled.sum(axis=1), axis=0) * 100



df_percentage['negative'] = df_percentage[1]
df_percentage['neutral'] = df_percentage[2] + df_percentage[3]
df_percentage['positive'] = df_percentage[4] + df_percentage[5]
#dfg = df_percentage[['negative', 'neutral', 'positive']]
#dfg = df_percentage[['neutral', 'positive']]
#dfg = df_percentage[['positive', 'neutral']]

# this
df_resampled['negative'] = df_resampled[1]
df_resampled['neutral'] = df_resampled[2] + df_resampled[3]
df_resampled['positive'] = df_resampled[4] + df_resampled[5]

dfg = df_resampled[['negative', 'neutral', 'positive']]
aggregated_data = dfg.sum()
total = aggregated_data.sum()

# Convert values to percentages
percentage_data = (aggregated_data / total) * 100

#aggregated_data = df_resampled.sum()

custom_labels = [
    f'Negative (1 star) - {int(aggregated_data["negative"])} Reviews',
    f'Neutral (2-3 stars) - {int(aggregated_data["neutral"])} Reviews',
    f'Positive (4-5 stars) - {int(aggregated_data["positive"])} Reviews',
                 ]

custom_labels = [
    f'Negative (1 star)',
    f'Neutral (2-3 stars)',
    f'Positive (4-5 stars)',
                 ]

custom_labels = percentage_data.values
custom_labels = ["94%",  "3%"    ,  "2.5%"]
custom_labels = ["3%"    ,  "2.5%"]
values = [27, 25]
colours = ['#a67c00', '#ffbf00']


custom_labels = ["3.1%"    ,  "4.1%"]
values = [30, 39]
colours = ['#7fa8ff', 'navy']

values = aggregated_data.values
values = [1948.90465762,   65.61822981,   85.47711257]
values = [1948.90465762,   65.61822981,   85.47711257]

# 'positive', 'neutral', 'negative
colours = ['#c4d7ff', '#7fa8ff','navy']
colours = ['#a67c00', '#ffbf00', '#ffdc73'][::-1]

fig = go.Figure(data=[go.Pie(labels=custom_labels, values=values, hole=.7, marker=dict(colors=colours))])
fig = go.Figure(data=[go.Pie(labels=custom_labels, values=values, hole=.7, marker=dict(colors=colours),  textinfo='label' )])

fig.update_layout(
    annotations=[dict(text='', x=0.5, y=0.5, font_size=40, showarrow=False)],
    title_font_size = 20,
    legend_font_size = 20,
    uniformtext_minsize=20,  # Minimum text size for all segments
    uniformtext_mode='hide'  # Hide text if it does not fit in a segment
)
fig.update_xaxes(title_font=dict(size=20))
fig.update_yaxes(title_font=dict(size=20))
fig.show()


#columns = dfg.columns

#columns = sorted(dfg.columns, reverse=True)
#columns = ['positive', 'neutral']
#dfg = df_percentage[columns]

dfg = df_percentage

columns = [1, 2, 3, 4, 5]

fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add bar for each rating
for rating in [1, 2, 3, 4, 5]:
    fig.add_trace(
        go.Bar(x=df_resampled.index, y=df_resampled[rating], name=f'Rating {rating}'),
        secondary_y=False,
    )

# Add line for 'neutral' and 'positive'
fig.add_trace(
    go.Scatter(x=df_percentage.index, y=df_percentage['neutral'], mode='lines+markers', name='Neutral', line=dict(dash='dot')),
    secondary_y=True,
)

fig.add_trace(
    go.Scatter(x=df_percentage.index, y=df_percentage['positive'], mode='lines+markers', name='Positive', line=dict(dash='dot')),
    secondary_y=True,
)

# fig.update_layout(
#     barmode='stack',
#     title='Stacked Bar Chart with Neutral and Positive Lines',
#     xaxis_title='Review Datetime',
#     yaxis_title='Percentage',
#     yaxis2_title='Neutral/Positive Percentage',
# )

dfg = df_resampled
xaxis_title = f'Date Review Made ({period_text})'

time_period = (dfg.index.max() - dfg.index.min()).days
num_reviews = len(df)

# Customize the layout
fig.update_layout(
    title={
        'text': f'{company_name_title} Review Ratings ({num_reviews} reviews over {time_period} days)',
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title=xaxis_title,
    yaxis_title='Rating Count',
    # xaxis={'type': 'category'},  # Treat x-axis values as discrete categories
    paper_bgcolor='rgb(255, 255, 255)',  # White
    plot_bgcolor='rgb(255, 255, 255)',  # White
    font=dict(
        family="Arial, sans-serif",
        size=16,
        color="black"
    ),
    xaxis=dict(
        showline=True,
        showgrid=False,
        linecolor='navy',
        type='category',
        tickangle=45,
        tickmode='auto',
        # nticks=20,
    ),
    yaxis=dict(
        showline=True,
        showgrid=True,
        gridcolor='teal',
        linecolor='navy'
    ),
    width=1000,
    bargap=0.4,
    showlegend=True,
)

# Show the figure
fig.show()


#=======================================================================================================================
# Output to file
#=======================================================================================================================

#base_filepath = "/Users/jhc/personal/mba/operations_management/group_assignment/data/data_with_datetime/
output_filename = f"{base_filepath}/{company_name}_trustpilot_{num_reviews}_reviews_{period}_data_overview.png"
width = 1920
height = 1080
pio.write_image(fig, output_filename, width=width, height=height, scale=2)
pio.write_image(fig, output_filename)
print(output_filename)




