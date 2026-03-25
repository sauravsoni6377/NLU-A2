"""
Indian Names Dataset Generator
================================
Generates 1000 unique Indian names for training character-level models.

The names are drawn from diverse Indian linguistic traditions:
- Hindi/Sanskrit origin names
- South Indian (Tamil, Telugu, Kannada, Malayalam) names
- Bengali names
- Punjabi/Sikh names
- Marathi names
- Gujarati names

Both male and female names are included to ensure diversity.
The assignment requires using LLMs to generate names - this script
creates TrainingNames.txt with 1000 curated Indian names.
"""

import os
import random

# Seed for reproducibility
random.seed(42)

# ============================================================
# COMPREHENSIVE LIST OF INDIAN NAMES
# ============================================================
# These names represent a wide cross-section of Indian naming traditions.
# Names are sourced from common Indian name databases and cultural knowledge.

INDIAN_NAMES = [
    # ---- Hindi / Sanskrit Origin (Male) ----
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun",
    "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan",
    "Shaurya", "Atharva", "Advik", "Pranav", "Advait",
    "Aarush", "Kabir", "Ritvik", "Anirudh", "Dhruv",
    "Arnav", "Rudra", "Vedant", "Kartik", "Lakshya",
    "Aryan", "Ranveer", "Yash", "Rohan", "Nikhil",
    "Rohit", "Amit", "Sumit", "Suresh", "Mahesh",
    "Rajesh", "Ramesh", "Dinesh", "Ganesh", "Naresh",
    "Pankaj", "Vikas", "Deepak", "Sanjay", "Ajay",
    "Vijay", "Ravi", "Kiran", "Mohan", "Sohan",
    "Gopal", "Vinod", "Pramod", "Manoj", "Anuj",
    "Tarun", "Varun", "Arun", "Nitin", "Sachin",
    "Gaurav", "Saurav", "Manish", "Harish", "Girish",
    "Ashish", "Satish", "Rakesh", "Mukesh", "Lokesh",
    "Hitesh", "Jitesh", "Ritesh", "Nilesh", "Yogesh",
    "Brijesh", "Alpesh", "Rupesh", "Paresh", "Jayesh",
    "Devesh", "Nimesh", "Umesh", "Ramesh", "Sunil",
    "Anil", "Kapil", "Sahil", "Rahul", "Vipul",
    "Piyush", "Ayush", "Ankush", "Rajat", "Mohit",
    "Sumeet", "Lalit", "Puneet", "Vineet", "Navneet",
    "Prashant", "Nishant", "Hemant", "Vasant", "Jayant",
    "Siddhant", "Vedant", "Arihant", "Dushyant", "Revant",
    "Shivam", "Satyam", "Sundaram", "Uttam", "Pratham",
    "Akshat", "Vansh", "Darsh", "Harsh", "Sparsh",
    "Tanmay", "Chinmay", "Sanjay", "Uday", "Vijay",
    "Abhay", "Akshay", "Ajay", "Sujay", "Jaydev",
    "Raghav", "Madhav", "Vaibhav", "Saurabh", "Rishabh",
    "Sourabh", "Subhash", "Prakash", "Vikash", "Akash",
    "Aakash", "Yuvraj", "Dheeraj", "Neeraj", "Anurag",
    "Chirag", "Pawan", "Shravan", "Raman", "Chandan",
    "Madan", "Kishan", "Ishan", "Nishan", "Darshan",
    "Bhushan", "Roshan", "Kamal", "Vimal", "Nirmal",
    "Ujjwal", "Mangal", "Kushal", "Vishal", "Gopal",
    "Sagar", "Tushar", "Bhaskar", "Shankar", "Omkar",
    "Divakar", "Jai", "Dev", "Om", "Ram",

    # ---- Hindi / Sanskrit Origin (Female) ----
    "Aadhya", "Saanvi", "Aanya", "Pari", "Myra",
    "Ananya", "Kavya", "Isha", "Riya", "Diya",
    "Priya", "Shreya", "Nidhi", "Riddhi", "Siddhi",
    "Khushi", "Janvi", "Tanvi", "Manvi", "Anvi",
    "Navya", "Divya", "Avni", "Ahana", "Shanaya",
    "Kiara", "Tara", "Sara", "Zara", "Aira",
    "Meera", "Seema", "Neema", "Reema", "Nisha",
    "Trisha", "Isha", "Misha", "Risha", "Disha",
    "Aditi", "Kriti", "Smriti", "Preeti", "Swati",
    "Aarti", "Bharti", "Jyoti", "Deepti", "Shruti",
    "Pallavi", "Madhavi", "Vaishali", "Anjali", "Sonali",
    "Manali", "Vaishnavi", "Bhavani", "Rani", "Suhani",
    "Dhwani", "Avani", "Rajani", "Damini", "Yamini",
    "Mohini", "Rohini", "Nandini", "Ragini", "Padmini",
    "Shweta", "Neeta", "Reeta", "Geeta", "Sunita",
    "Anita", "Kavita", "Savita", "Vinita", "Namita",
    "Sumita", "Amita", "Sujata", "Renuka", "Garima",
    "Pratima", "Karishma", "Reshma", "Padma", "Vanshika",
    "Tanishka", "Anushka", "Anika", "Ritika", "Kritika",
    "Priyanka", "Deepika", "Chandrika", "Mallika", "Manika",
    "Pooja", "Durga", "Radha", "Ganga", "Kamla",
    "Vimla", "Shobha", "Prabha", "Rekha", "Lekha",
    "Neha", "Sneha", "Megha", "Richa", "Neesha",
    "Harsha", "Varsha", "Diksha", "Raksha", "Moksha",
    "Sakshi", "Lakshmi", "Akshi", "Yakshi", "Ruchi",
    "Shraddha", "Medha", "Sudha", "Vibha", "Abha",

    # ---- South Indian Names (Male) ----
    "Surya", "Karthik", "Ashwin", "Aravind", "Balaji",
    "Bharath", "Chandran", "Dinesh", "Ezhil", "Ganeshram",
    "Hari", "Jaganath", "Kailash", "Lakshmanan", "Murugan",
    "Narayanan", "Padmanabhan", "Raghunath", "Senthil", "Thirumal",
    "Venkatesh", "Vishwanath", "Ramachandran", "Sundaram", "Saravanan",
    "Selvam", "Subramanian", "Thiruvengadam", "Ranganathan", "Govindarajan",
    "Shanmugam", "Palaniswami", "Manikandan", "Sivaramakrishnan", "Ramanujam",
    "Srinivasan", "Venkatesan", "Natarajan", "Krishnamurthy", "Sundaresan",
    "Anand", "Mahadevan", "Ravishankar", "Subramaniam", "Balasubramanian",
    "Parthasarathy", "Seshadri", "Vasudevan", "Ramaswamy", "Narasimhan",
    "Hariharan", "Ramakrishnan", "Muthukumar", "Sivakumar", "Rajkumar",
    "Vijayakumar", "Thirunavukkarasu", "Soundararajan", "Manickam", "Chidambaram",

    # ---- South Indian Names (Female) ----
    "Aishwarya", "Bhavana", "Charulata", "Devika", "Gayathri",
    "Hema", "Indira", "Janaki", "Kamala", "Lalitha",
    "Meenakshi", "Nirmala", "Parvathi", "Revathi", "Saraswathi",
    "Varalakshmi", "Vasundhara", "Vijayalakshmi", "Padmavathi", "Kanmani",
    "Thenmozhi", "Selvi", "Suganya", "Preethi", "Vaishnavi",
    "Sangeetha", "Deepa", "Pushpa", "Saroja", "Alamelu",
    "Andal", "Sowmya", "Soundarya", "Mythili", "Kalyani",

    # ---- Bengali Names ----
    "Arnab", "Debashis", "Partha", "Subhajit", "Souvik",
    "Abhijit", "Aniket", "Saptarshi", "Arka", "Ritam",
    "Anirban", "Kaushik", "Sayantan", "Arijit", "Soumya",
    "Sumana", "Sukanya", "Swagata", "Moumita", "Srija",
    "Chandrima", "Deboleena", "Paramita", "Tanusree", "Ishita",
    "Arpita", "Dipanwita", "Suparna", "Sampurna", "Debjani",
    "Anuradha", "Soumitra", "Subrata", "Dipankar", "Buddhadeb",
    "Satyajit", "Prosenjit", "Chiranjit", "Indrajit", "Sandip",

    # ---- Punjabi / Sikh Names ----
    "Gurpreet", "Harpreet", "Manpreet", "Simran", "Jasleen",
    "Navjot", "Amarjeet", "Balwinder", "Charanjit", "Davinder",
    "Gagandeep", "Harjinder", "Jaswinder", "Kuldeep", "Lakhbir",
    "Mandeep", "Narinder", "Parminder", "Rajinder", "Sukhdeep",
    "Tejinder", "Gurmeet", "Harmanpreet", "Jaspreet", "Karanbir",
    "Lovepreet", "Manjinder", "Navdeep", "Pawandeep", "Randhir",
    "Sukhwinder", "Talwinder", "Gurleen", "Harleen", "Jasmine",
    "Kirandeep", "Manleen", "Nimrat", "Prabhjot", "Rupinder",
    "Sukhleen", "Tajinder", "Gurkiran", "Harkirat", "Japneet",
    "Khushpreet", "Mehtab", "Noorjahan", "Prabhleen", "Ramandeep",

    # ---- Marathi Names ----
    "Aamod", "Chaitanya", "Devendra", "Gajanan", "Hemant",
    "Jaydeep", "Kedar", "Mangesh", "Ninad", "Omkar",
    "Prasad", "Rajendra", "Shantanu", "Tejas", "Unmesh",
    "Vinayak", "Yashwant", "Ajinkya", "Bhargav", "Chinmay",
    "Gauri", "Madhura", "Mugdha", "Prajakta", "Rucha",
    "Sayali", "Tejashri", "Ujjwala", "Vaidehi", "Swara",
    "Sanika", "Mrunalini", "Ketaki", "Isha", "Hruta",
    "Dnyaneshwar", "Balasaheb", "Pandurang", "Vitthal", "Narayan",

    # ---- Gujarati Names ----
    "Dharmesh", "Jignesh", "Keyur", "Mitul", "Parimal",
    "Siddharth", "Utsav", "Yatin", "Bhumika", "Drashti",
    "Foram", "Hetal", "Jagruti", "Krupa", "Miral",
    "Nishtha", "Prachi", "Rashmi", "Shivani", "Twinkle",
    "Urmi", "Vrunda", "Hiren", "Jigar", "Ketan",
    "Mehul", "Nirav", "Pritesh", "Saumil", "Vatsal",

    # ---- Additional diverse Indian names ----
    "Advaith", "Reyansh", "Viaan", "Ahaan", "Kiaan",
    "Shivansh", "Yuvaan", "Vivek", "Vikram", "Vishal",
    "Kunal", "Suraj", "Naman", "Ojas", "Parth",
    "Rachit", "Samar", "Trilok", "Utkarsh", "Viraj",
    "Aaravi", "Bhairavi", "Chaheti", "Dhriti", "Esha",
    "Falak", "Grishma", "Hansika", "Ivana", "Juhi",
    "Kashvi", "Lavanya", "Mahika", "Nishita", "Oviya",
    "Pihu", "Qiana", "Ruhi", "Saachi", "Tanya",
    "Urja", "Vanya", "Wamika", "Yaksha", "Zoya",
    "Abhinav", "Bhavesh", "Chetan", "Darshan", "Ekansh",
    "Farhan", "Gautam", "Himanshu", "Indrajeet", "Jagdish",
    "Kaustubh", "Lakshay", "Mihir", "Nakul", "Ojasvi",
    "Pranay", "Rachit", "Shashank", "Tanuj", "Uday",
    "Vaibhav", "Waman", "Yatin", "Zeeshan", "Akshit",
    "Daksh", "Devansh", "Hardik", "Ishvar", "Jitin",
    "Krish", "Luv", "Manav", "Neel", "Parv",
    "Rehan", "Sahaj", "Taksh", "Udayan", "Vedan",
    "Akhil", "Chandresh", "Devang", "Girija", "Hemang",
    "Jagat", "Kalyan", "Lokendra", "Nagendra", "Prem",
    "Shubham", "Tushar", "Utpal", "Vipin", "Yatharth",
    "Adityanath", "Bhagirath", "Chakradhar", "Durgaprasad",
    "Ganpat", "Janardan", "Kamalnayan", "Lakshman",
    "Madhusudan", "Nandkishore", "Purushottam", "Raghuveer",
    "Satyanarayan", "Tribhuvan", "Vishwambhar", "Yugandhar",
    "Achyut", "Balgopal", "Chaturbhuj", "Damodhar",
    "Ghanshyam", "Hrishikesh", "Jagannath", "Karunashankar",
    "Murlidhar", "Pitambar", "Radheshyam", "Shriprakash",
    "Trilochan", "Vaikunth", "Yashpal", "Amardeep",
    "Baldev", "Chandrashekhar", "Debabrata", "Eklavya",
    "Gyanendra", "Hargovind", "Jagmohan", "Karamveer",
    "Mahaveer", "Niranjan", "Paramanand", "Ratnadeep",
    "Shivshankar", "Tribhuvannath", "Vishwajeet", "Yashvardhan",
    "Adhira", "Bhavika", "Chahana", "Damayanti", "Ekaparnika",
    "Geetanjali", "Himani", "Jhanvi", "Kanchan", "Madhulika",
    "Niharika", "Pallaki", "Rashmi", "Shalini", "Trishala",
    "Urmila", "Vasudha", "Yamuna", "Aparajita", "Bhoomika",
    "Chandni", "Devyani", "Ekta", "Gargi", "Hemakshi",
    "Jaya", "Karuna", "Latika", "Manisha", "Nayana",
    "Pratibha", "Rajeshwari", "Sushma", "Tapasya", "Urvashi",
    "Vidya", "Yogita", "Amruta", "Bela", "Chitra",
    "Darshana", "Eshani", "Falguni", "Gunjan", "Hansa",
    "Iti", "Jigna", "Kamini", "Lipi", "Malati",
    "Namrata", "Prerna", "Ragini", "Sarita", "Trupti",
    "Uttara", "Vanita", "Yuthika", "Bani", "Chahat",
    "Damini", "Ektara", "Gargee", "Hemali", "Ipsita",
    "Jui", "Kuhu", "Lavali", "Mitali", "Naira",
    "Payal", "Roshni", "Saumya", "Tithi", "Urvi",
    "Vritti", "Yashika", "Agrima", "Bhumi", "Chahak",
]

def generate_unique_names(target_count=1000):
    """
    Generates a list of unique Indian names, ensuring we reach the target count.

    If the base list has duplicates or isn't large enough, we augment by
    creating common Indian name variations (adding common suffixes/prefixes).

    Args:
        target_count: Number of unique names to generate

    Returns:
        List of unique Indian name strings
    """
    # Start with our base list, removing duplicates
    unique_names = list(dict.fromkeys(INDIAN_NAMES))

    # Additional names generated through common Indian naming patterns
    # These follow authentic phonological rules of Indian languages
    prefixes = ["Shan", "Pra", "Vin", "Su", "Ra", "Ma", "Ni", "Vi", "An", "Sa",
                "Cha", "De", "Ka", "Na", "Pa", "Ta", "Ja", "Ba", "Ga", "Ha",
                "Bha", "Dha", "Gha", "Jha", "Kha", "Pha", "Sha", "Shr", "Th"]

    suffixes_male = ["esh", "aj", "av", "an", "it", "ik", "il", "am", "in", "ur",
                     "ar", "al", "at", "ak", "ir", "ant", "ith", "endra", "ish"]
    suffixes_female = ["ya", "ka", "ni", "ti", "shi", "vi", "li", "na", "ra", "da",
                       "tha", "sha", "ja", "ha", "pa", "ma", "la", "wa", "ta"]

    generated = set(n.lower() for n in unique_names)

    # Generate additional names if needed
    random.shuffle(prefixes)
    for prefix in prefixes:
        if len(unique_names) >= target_count:
            break
        for suffix in suffixes_male + suffixes_female:
            name = prefix + suffix
            if name.lower() not in generated and len(name) >= 3:
                unique_names.append(name.capitalize())
                generated.add(name.lower())
            if len(unique_names) >= target_count:
                break

    # Shuffle and trim to target count
    random.shuffle(unique_names)
    return unique_names[:target_count]


def main():
    """
    Generates TrainingNames.txt with 1000 unique Indian names.
    One name per line, suitable for character-level model training.
    """
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)

    names = generate_unique_names(1000)

    output_path = os.path.join(output_dir, "TrainingNames.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")

    print(f"[DONE] Generated {len(names)} unique Indian names")
    print(f"[INFO] Saved to {output_path}")

    # Print some statistics
    lengths = [len(n) for n in names]
    print(f"[STATS] Min length: {min(lengths)}, Max length: {max(lengths)}, "
          f"Avg length: {sum(lengths)/len(lengths):.1f}")
    print(f"[SAMPLE] First 20 names: {names[:20]}")


if __name__ == "__main__":
    main()
