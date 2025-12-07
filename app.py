# app.py
import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from passlib.hash import bcrypt
from datetime import datetime
import json
import os

# ---------------------------
# CONFIG & DATA (expanded)
# ---------------------------
DB_PATH = "skill_analyzer.db"

JOB_ROLE_TEMPLATES = {
    "AWS Solutions Architect": (
        "aws cloud architecture vpc ec2 s3 iam eks ecs terraform cloudformation "
        "networking security scalability cost-optimization load-balancer autoscaling "
        "monitoring logging observability"
    ),
    "Data Scientist": (
        "python pandas numpy scikit-learn statistics machine-learning deep-learning "
        "tensorflow pytorch data-visualization feature-engineering model-evaluation "
        "nlp time-series"
    ),
    "Full Stack Developer": (
        "javascript typescript react angular vue nodejs express html css sql nosql "
        "rest api graphql docker testing ci/cd security web-performance"
    ),
    "DevOps Engineer": (
        "ci/cd jenkins github-actions gitlab-ci terraform ansible kubernetes docker "
        "monitoring logging prometheus grafana automation scripting linux networking"
    ),
    "Machine Learning Engineer": (
        "python mlops model-deployment docker kubernetes tensorflow pytorch "
        "feature-store data-pipelines spark airflow optimization inference serving"
    ),
    "Product Manager": (
        "product-management roadmap stakeholder-communication metrics a-b-testing "
        "prioritization user-research ux strategy go-to-market analytics"
    ),
    "Data Engineer": (
        "etl pipelines spark airflow kafka data-warehousing redshift bigquery sql nosql "
        "parquet avro schema-design performance tuning"
    )
}

# Course catalog with roadmap details
COURSE_CATALOG = [
    {
        "title": "AWS Certified Solutions Architect - Fundamentals",
        "skills": "aws ec2 vpc s3 iam cloudformation terraform autoscaling load-balancer monitoring",
        "roadmap": {
            "prerequisites": ["Basic Linux", "Basic networking (TCP/IP, CIDR)"],
            "duration": "4-6 weeks (part-time)",
            "steps": [
                "Intro to AWS core services (EC2, S3, VPC, IAM)",
                "Hands-on: Launch EC2, configure VPC and subnets",
                "Study IAM roles, policies and secure access",
                "Learn load balancing and auto-scaling groups",
                "Infrastructure as Code: basics of CloudFormation & Terraform",
                "Monitoring and cost optimisation basics"
            ],
            "next_steps": ["Architectural best practices", "Specialize in security or networking"]
        }
    },
    {
        "title": "Intro to Data Science with Python",
        "skills": "python pandas numpy data-visualization scikit-learn statistics",
        "roadmap": {
            "prerequisites": ["Basic Python"],
            "duration": "3-5 weeks",
            "steps": [
                "Python essentials refresher (data structures, functions)",
                "Pandas for data wrangling",
                "NumPy for numerical computing",
                "Exploratory Data Analysis & visualization with matplotlib/seaborn",
                "Intro to scikit-learn and basic ML models",
                "Model evaluation and feature selection"
            ],
            "next_steps": ["Advanced ML", "Deep Learning", "MLOps basics"]
        }
    },
    {
        "title": "Full-Stack Web Development Bootcamp",
        "skills": "javascript react nodejs express html css sql rest api docker ci/cd testing",
        "roadmap": {
            "prerequisites": ["Basic programming skills"],
            "duration": "8-10 weeks",
            "steps": [
                "HTML & CSS fundamentals",
                "JavaScript fundamentals & DOM",
                "Frontend frameworks: React basics",
                "Backend with Node.js and Express",
                "Database basics (SQL/NoSQL)",
                "Build & deploy a full-stack app, add tests and CI/CD"
            ],
            "next_steps": ["Advanced React patterns", "Microservices & cloud deployment"]
        }
    },
    {
        "title": "Kubernetes for Developers",
        "skills": "kubernetes docker pods services helm deployment ingress monitoring",
        "roadmap": {
            "prerequisites": ["Docker basics", "Linux command line"],
            "duration": "3-4 weeks",
            "steps": [
                "Docker image creation & registries",
                "Kubernetes primitives: pods, deployments, services",
                "ConfigMaps and Secrets",
                "Helm basics and templating",
                "Ingress, networking and service mesh overview",
                "Monitoring and troubleshooting (kubectl, logs, metrics)"
            ],
            "next_steps": ["Production-grade clusters", "Kubernetes security"]
        }
    },
    {
        "title": "MLOps: Productionizing Machine Learning",
        "skills": "mlops model-deployment monitoring pipelines airflow docker ci/cd feature-store",
        "roadmap": {
            "prerequisites": ["Basic ML model knowledge", "Python"],
            "duration": "4-6 weeks",
            "steps": [
                "ML lifecycle and reproducibility",
                "Model packaging and containerization",
                "CI/CD for ML pipelines",
                "Deploying models as services",
                "Monitoring models and data drift detection",
                "Automating retraining and lineage tracking"
            ],
            "next_steps": ["Feature store design", "Advanced monitoring & governance"]
        }
    },
    {
        "title": "DevOps and CI/CD with Jenkins & GitHub Actions",
        "skills": "ci/cd jenkins github-actions docker pipeline automation terraform",
        "roadmap": {
            "prerequisites": ["Git basics", "Scripting basics"],
            "duration": "2-4 weeks",
            "steps": [
                "Version control workflows and branching",
                "Jenkins pipelines and jobs (declarative & scripted)",
                "GitHub Actions workflows and secrets",
                "Automated testing and artifact publishing",
                "Deployment orchestration and rollbacks",
                "Integrate with Terraform for infra pipelines"
            ],
            "next_steps": ["Security scanning in pipelines", "Infrastructure testing"]
        }
    },
    {
        "title": "System Design Basics",
        "skills": "scalability microservices load-balancing caching databases reliability",
        "roadmap": {
            "prerequisites": ["Basic web/backend knowledge"],
            "duration": "3-6 weeks",
            "steps": [
                "Learn scalability patterns and caching",
                "Designing for reliability and availability",
                "Database scaling: sharding, replication",
                "Load balancing and async processing",
                "Data modeling and consistency tradeoffs",
                "Practice common design problems (URL shortener, chat, etc.)"
            ],
            "next_steps": ["Advanced distributed systems", "Cloud architecture patterns"]
        }
    },
    {
        "title": "Data Engineering with Spark & Airflow",
        "skills": "spark airflow etl pipelines kafka data-warehousing sql parquet",
        "roadmap": {
            "prerequisites": ["SQL basics", "Python"],
            "duration": "4-6 weeks",
            "steps": [
                "ETL fundamentals and data ingestion",
                "Intro to Apache Spark dataframes",
                "Batch processing patterns",
                "Orchestration with Airflow",
                "Streaming basics with Kafka",
                "Data warehouse design and storage formats (Parquet, ORC)"
            ],
            "next_steps": ["Cloud data platforms", "Data lakehouse concepts"]
        }
    }
]

# ---------------------------
# DATABASE (unchanged)
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    # users
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            name TEXT,
            password_hash TEXT,
            created_at TEXT
        )
    ''')
    # profile: basic info + skills stored as JSON list
    c.execute('''
        CREATE TABLE IF NOT EXISTS profiles (
            user_id INTEGER PRIMARY KEY,
            headline TEXT,
            education TEXT,
            experience TEXT,
            skills_json TEXT,
            certifications_json TEXT,
            learning_history_json TEXT,
            progress INTEGER DEFAULT 0,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    # badges
    c.execute('''
        CREATE TABLE IF NOT EXISTS badges (
            user_id INTEGER,
            badge TEXT,
            awarded_at TEXT
        )
    ''')
    # leaderboard (simple)
    c.execute('''
        CREATE TABLE IF NOT EXISTS leaderboard (
            user_id INTEGER PRIMARY KEY,
            score INTEGER DEFAULT 0,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    return conn

conn = init_db()

# ---------------------------
# AUTH
# ---------------------------
def create_user(email, name, password):
    c = conn.cursor()
    pwd_hash = bcrypt.hash(password)
    try:
        c.execute("INSERT INTO users (email, name, password_hash, created_at) VALUES (?, ?, ?, ?)",
                  (email.lower(), name, pwd_hash, datetime.utcnow().isoformat()))
        conn.commit()
        user_id = c.lastrowid
        # initialize empty profile and leaderboard
        c.execute("INSERT OR REPLACE INTO profiles (user_id, skills_json, learning_history_json) VALUES (?, ?, ?)",
                  (user_id, json.dumps([]), json.dumps([])))
        c.execute("INSERT OR IGNORE INTO leaderboard (user_id, score) VALUES (?, ?)", (user_id, 0))
        conn.commit()
        return user_id
    except sqlite3.IntegrityError:
        return None

def verify_user(email, password):
    c = conn.cursor()
    c.execute("SELECT id, password_hash, name FROM users WHERE email = ?", (email.lower(),))
    row = c.fetchone()
    if not row:
        return None
    uid, pwd_hash, name = row
    try:
        if bcrypt.verify(password, pwd_hash):
            return {"id": uid, "email": email.lower(), "name": name}
    except Exception:
        return None
    return None

# ---------------------------
# PROFILE Helpers
# ---------------------------
def get_profile(user_id):
    c = conn.cursor()
    c.execute("SELECT headline, education, experience, skills_json, certifications_json, learning_history_json, progress FROM profiles WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    if not row:
        return None
    headline, education, experience, skills_json, certifications_json, learning_history_json, progress = row
    return {
        "headline": headline or "",
        "education": education or "",
        "experience": experience or "",
        "skills": json.loads(skills_json) if skills_json else [],
        "certifications": json.loads(certifications_json) if certifications_json else [],
        "learning_history": json.loads(learning_history_json) if learning_history_json else [],
        "progress": progress or 0
    }

def save_profile(user_id, headline, education, experience, skills_list, certifications_list):
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO profiles (user_id, headline, education, experience, skills_json, certifications_json)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, headline, education, experience, json.dumps(skills_list), json.dumps(certifications_list)))
    conn.commit()

def update_learning_history(user_id, course_title):
    profile = get_profile(user_id)
    history = profile.get("learning_history", [])
    history.append({"course": course_title, "date": datetime.utcnow().isoformat()})
    c = conn.cursor()
    c.execute("UPDATE profiles SET learning_history_json = ? WHERE user_id = ?", (json.dumps(history), user_id))
    # increment progress and leaderboard score
    c.execute("UPDATE profiles SET progress = progress + 10 WHERE user_id = ?", (user_id,))
    c.execute("UPDATE leaderboard SET score = score + 10 WHERE user_id = ?", (user_id,))
    conn.commit()

def award_badge(user_id, badge):
    c = conn.cursor()
    c.execute("INSERT INTO badges (user_id, badge, awarded_at) VALUES (?, ?, ?)", (user_id, badge, datetime.utcnow().isoformat()))
    conn.commit()

def get_badges(user_id):
    c = conn.cursor()
    c.execute("SELECT badge, awarded_at FROM badges WHERE user_id = ?", (user_id,))
    rows = c.fetchall()
    return [{"badge": r[0], "awarded_at": r[1]} for r in rows]

# ---------------------------
# CORE "AI" LOGIC
# ---------------------------
def analyze_skill_gaps(user_skills, target_role=None, top_k=8):
    if not user_skills:
        user_text = ""
    else:
        user_text = " ".join(user_skills).lower()
    templates = JOB_ROLE_TEMPLATES if target_role is None else {target_role: JOB_ROLE_TEMPLATES[target_role]}
    results = []
    for role, role_txt in templates.items():
        try:
            vectorizer = TfidfVectorizer().fit([user_text, role_txt])
            vecs = vectorizer.transform([user_text, role_txt])
            sim = cosine_similarity(vecs[0], vecs[1])[0][0]
        except Exception:
            sim = 0.0
        role_tokens = set(role_txt.split())
        user_tokens = set(user_text.split())
        missing = list(role_tokens - user_tokens)
        results.append({
            "role": role,
            "similarity": float(sim),
            "missing_skills": missing[:top_k]
        })
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    return results

def recommend_courses(missing_skills, top_n=6):
    recs = []
    if not missing_skills:
        return []
    for course in COURSE_CATALOG:
        course_skills = set(course["skills"].split())
        overlap = len(set(missing_skills) & course_skills)
        score = overlap
        if score > 0:
            recs.append((score, course))
    recs = sorted(recs, key=lambda x: x[0], reverse=True)
    return [c for s,c in recs][:top_n]

def suggest_mentors(user_skills, top_n=3):
    user_set = set([s.lower() for s in user_skills]) if user_skills else set()
    # simple overlap heuristic
    mentor_scores = []
    for m in [
        {"name":"Asha R","role":"Senior Data Scientist","skills":"python pandas scikit-learn ml model-deployment"},
        {"name":"Rahul K","role":"DevOps Lead","skills":"kubernetes terraform ci/cd monitoring docker"},
        {"name":"Meera S","role":"AWS Architect","skills":"aws vpc iam s3 cost-optimization"},
        {"name":"Vikram P","role":"Full-Stack Eng","skills":"react nodejs docker sql api-design"}
    ]:
        overlap = len(user_set & set(m["skills"].split()))
        mentor_scores.append((overlap, m))
    mentor_scores = sorted(mentor_scores, key=lambda x: x[0], reverse=True)
    return [m for s,m in mentor_scores if s>0][:top_n] or [m for s,m in mentor_scores][:top_n]

def suggest_projects(user_skills, top_n=3):
    user_set = set(user_skills) if user_skills else set()
    project_scores = []
    for pj in [
        {"title":"E-commerce Traffic Forecasting","required_skills":"python pandas time-series ml"},
        {"title":"Microservices Migration to K8s","required_skills":"kubernetes docker terraform ci/cd"},
        {"title":"Cloud Cost Optimization","required_skills":"aws monitoring cost-optimization terraform"}
    ]:
        overlap = len(user_set & set(pj["required_skills"].split()))
        project_scores.append((overlap, pj))
    project_scores = sorted(project_scores, key=lambda x: x[0], reverse=True)
    return [p for s,p in project_scores if s>0][:top_n] or [p for s,p in project_scores][:top_n]

def career_forecast(user_skills):
    # basic mock: role -> 12 month trending numbers based on similarity
    role_scores = []
    for role, template in JOB_ROLE_TEMPLATES.items():
        try:
            sim = cosine_similarity(TfidfVectorizer().fit_transform([" ".join(user_skills), template]))[0,1]
        except Exception:
            sim = 0.0
        months = list(range(12))
        base = sim * 50
        noise = np.random.randn(12) * 2
        trend = [max(0, base + m * (sim*2) + float(noise[i])) for i,m in enumerate(months)]
        role_scores.append((role, list(map(float, trend))))
    return role_scores

# ---------------------------
# Helpers for courses & roadmap
# ---------------------------
def get_course_by_title(title):
    for c in COURSE_CATALOG:
        if c["title"] == title:
            return c
    return None

# ---------------------------
# UI / PAGES (no images)
# ---------------------------
def login_page():
    st.title("AI Career GPS — Login or Sign up")
    st.write("AI-Driven Skill Gap Analyzer.")

    menu = ["Login", "Sign up"]
    choice = st.radio("Choose action", menu)

    if choice == "Sign up":
        st.subheader("Create new account")
        name = st.text_input("Full name")
        email = st.text_input("Email")
        pwd = st.text_input("Password", type="password")
        pwd2 = st.text_input("Confirm password", type="password")
        if st.button("Create account"):
            if not (name and email and pwd and pwd2):
                st.error("Please fill all fields")
            elif pwd != pwd2:
                st.error("Passwords do not match")
            else:
                uid = create_user(email, name, pwd)
                if uid:
                    st.success("Account created — you can login now")
                else:
                    st.error("Account with this email already exists")

    elif choice == "Login":
        st.subheader("Login")
        email = st.text_input("Email", key="login_email")
        pwd = st.text_input("Password", type="password", key="login_pwd")
        if st.button("Login"):
            user = verify_user(email, pwd)
            if user:
                st.success(f"Welcome back, {user['name']}")
                st.session_state["user"] = user
            else:
                st.error("Login failed — check email/password")

def profile_page():
    st.title("Your Profile")
    user = st.session_state.get("user")
    if not user:
        st.error("You must be logged in")
        return
    profile = get_profile(user["id"])
    if profile is None:
        st.error("Profile not found")
        return

    st.write(f"**Name:** {user['name']}")
    st.write(f"**Email:** {user['email']}")

    with st.form("profile_form"):
        headline = st.text_input("Professional headline", value=profile.get("headline", ""))
        education = st.text_input("Education", value=profile.get("education", ""))
        experience = st.text_area("Experience / Summary", value=profile.get("experience", ""))
        skills_text = st.text_input("Skills (comma-separated)", value=", ".join(profile.get("skills", [])))
        certifications_text = st.text_input("Certifications (comma-separated)", value=", ".join(profile.get("certifications", [])))
        submitted = st.form_submit_button("Save profile")
        if submitted:
            skills_list = [s.strip().lower() for s in skills_text.split(",") if s.strip()]
            cert_list = [c.strip() for c in certifications_text.split(",") if c.strip()]
            save_profile(user["id"], headline, education, experience, skills_list, cert_list)
            st.success("Profile saved")
            award_badge(user["id"], "Profile completed")

def home_dashboard():
    st.title("Home — Dashboard")
    user = st.session_state.get("user")
    if not user:
        st.error("Please login")
        return
    profile = get_profile(user["id"])
    if profile is None:
        st.info("No profile yet. Please fill your profile.")
        return

    # Top metrics (textual, no images)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Progress", f"{profile.get('progress', 0)}%")
    with col2:
        badges = get_badges(user["id"])
        st.metric("Badges Earned", len(badges))
    with col3:
        analysis = analyze_skill_gaps(profile.get("skills", []))
        if analysis:
            st.metric("Top matched role", analysis[0]["role"], delta=f"score {analysis[0]['similarity']:.2f}")
        else:
            st.write("Add skills in Profile to get role matches.")

    st.markdown("---")
    st.subheader("Quick Actions")
    if st.button("Run Skill Analysis"):
        st.session_state["analysis_trigger"] = True
        st.rerun()
    if st.button("View Recommendations"):
        st.session_state["view_recs"] = True
        st.rerun()

    st.markdown("---")
    st.subheader("Recent Learning History")
    hist = profile.get("learning_history", [])
    if hist:
        df = pd.DataFrame(hist)[["date","course"]]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        st.table(df.tail(6))
    else:
        st.write("No learning history yet. Run Skill Analyzer for recommendations and mark courses complete.")

def skill_analyzer_page():
    st.title("Skill Analyzer")
    user = st.session_state.get("user")
    if not user:
        st.error("Please login")
        return
    profile = get_profile(user["id"])
    if profile is None:
        st.info("No profile yet. Please fill your profile.")
        return

    st.subheader("Enter or update your skills")
    skills_text = st.text_input("Type your skills (comma separated)", value=", ".join(profile.get("skills", [])))
    role_choice = st.selectbox("Compare against role template (optional)", ["--none--"] + list(JOB_ROLE_TEMPLATES.keys()))
    if st.button("Analyze"):
        skills_list = [s.strip().lower() for s in skills_text.split(",") if s.strip()]
        save_profile(user["id"], profile.get("headline",""), profile.get("education",""), profile.get("experience",""), skills_list, profile.get("certifications",[]))
        analysis = analyze_skill_gaps(skills_list, None if role_choice=="--none--" else role_choice)
        st.success("Analysis complete")
        for a in analysis:
            st.markdown(f"### Role: {a['role']}")
            st.write(f"Similarity score: **{a['similarity']:.3f}**")
            if a['missing_skills']:
                st.write("Suggested skills to learn:", ", ".join(a['missing_skills']))
                recs = recommend_courses(a['missing_skills'])
                if recs:
                    st.write("Recommended courses:")
                    for r in recs:
                        st.write(f"- **{r['title']}** — covers: {r['skills']}")
                        # roadmap expander
                        c = get_course_by_title(r['title'])
                        if c:
                            with st.expander(f"Show roadmap for '{c['title']}'"):
                                st.write("**Prerequisites:**", ", ".join(c["roadmap"]["prerequisites"]))
                                st.write("**Estimated duration:**", c["roadmap"]["duration"])
                                st.write("**Steps:**")
                                for i, step in enumerate(c["roadmap"]["steps"], start=1):
                                    st.write(f"{i}. {step}")
                                st.write("**Next steps after completion:**", ", ".join(c["roadmap"]["next_steps"]))
                                if st.button(f"Mark '{c['title']}' as completed (Analyze page)", key=f"comp_an_{c['title']}"):
                                    update_learning_history(user["id"], c['title'])
                                    award_badge(user["id"], "Course completed")
                                    st.success(f"Marked '{c['title']}' completed")
                else:
                    st.write("No local course matches found.")
            else:
                st.write("No major missing skills detected for this role — good match!")

        # Mentor & project suggestions
        st.markdown("---")
        st.subheader("Mentor suggestions")
        mentors = suggest_mentors(skills_list)
        if mentors:
            for m in mentors:
                st.write(f"- **{m['name']}** ({m['role']}) — skills: {m['skills']}")
        else:
            st.write("No mentors found for current skills.")

        st.subheader("Project suggestions")
        projects = suggest_projects(skills_list)
        for p in projects:
            st.write(f"- **{p['title']}** — needs: {p['required_skills']}")

def recommendations_page():
    st.title("Personalized Recommendations")
    user = st.session_state.get("user")
    if not user:
        st.error("Please login")
        return
    profile = get_profile(user["id"])
    if profile is None:
        st.info("No profile yet. Please fill your profile.")
        return
    skills = profile.get("skills", [])
    if not skills:
        st.info("No skills set — go to Profile and update your skills first.")
        return

    analysis = analyze_skill_gaps(skills)
    best = analysis[0]
    st.markdown(f"**Best matched role:** {best['role']}  •  **Score:** {best['similarity']:.2f}")
    st.write("**Missing skills:**", ", ".join(best["missing_skills"]) or "None")

    rec_courses = recommend_courses(best["missing_skills"])
    st.subheader("Courses to fill the gap")
    if rec_courses:
        for c in rec_courses:
            st.write(f"- **{c['title']}** — covers: {c['skills']}")
            course = get_course_by_title(c['title'])
            if course:
                with st.expander(f"Roadmap: {course['title']}"):
                    st.write("**Prerequisites:**", ", ".join(course["roadmap"]["prerequisites"]))
                    st.write("**Estimated duration:**", course["roadmap"]["duration"])
                    st.write("**Steps:**")
                    for idx, step in enumerate(course["roadmap"]["steps"], start=1):
                        st.write(f"{idx}. {step}")
                    st.write("**Next steps:**", ", ".join(course["roadmap"]["next_steps"]))
                    if st.button(f"Mark '{course['title']}' as completed", key=f"rc_{course['title']}"):
                        update_learning_history(user["id"], course['title'])
                        award_badge(user["id"], "Course completed")
                        st.success("Marked as completed and progress updated")
    else:
        st.write("No direct local course matches — try other learning platforms.")

    st.markdown("---")
    st.subheader("Mentors you can reach out to")
    for m in suggest_mentors(skills):
        st.write(f"- {m['name']} — {m['role']} (skills: {m['skills']})")

    st.markdown("---")
    st.subheader("Project opportunities")
    for p in suggest_projects(skills):
        st.write(f"- {p['title']} — required: {p['required_skills']}")

def leaderboard_page():
    st.title("Leaderboard")
    c = conn.cursor()
    df = pd.read_sql_query("""
        SELECT u.name, l.score FROM leaderboard l
        JOIN users u ON u.id = l.user_id
        ORDER BY l.score DESC LIMIT 10
    """, conn)
    if df.empty:
        st.write("No scores yet — start learning to appear here!")
    else:
        st.table(df)

def logout():
    if "user" in st.session_state:
        del st.session_state["user"]
    st.success("Logged out")
    st.rerun()

# ---------------------------
# APP START
# ---------------------------
def main():
    st.set_page_config(page_title="AI Career GPS", layout="wide")

    if "user" not in st.session_state:
        st.session_state["user"] = None
    if "analysis_trigger" not in st.session_state:
        st.session_state["analysis_trigger"] = False
    if "view_recs" not in st.session_state:
        st.session_state["view_recs"] = False

    st.sidebar.title("AI Career GPS")
    menu = ["Home", "Profile", "Skill Analyzer", "Recommendations", "Leaderboard", "Logout"]

    if st.session_state["user"] is None:
        st.sidebar.info("Please login or sign up to continue")
        login_page()
        return

    user = st.session_state.get("user")
    display_name = user.get("name") if isinstance(user, dict) and user.get("name") else user.get("email")
    st.sidebar.write(f"Signed in as: **{display_name}**")
    choice = st.sidebar.radio("Go to", menu)

    if choice == "Home":
        home_dashboard()
    elif choice == "Profile":
        profile_page()
    elif choice == "Skill Analyzer":
        skill_analyzer_page()
    elif choice == "Recommendations":
        recommendations_page()
    elif choice == "Leaderboard":
        leaderboard_page()
    elif choice == "Logout":
        logout()
        return

if __name__ == "__main__":
    main()
