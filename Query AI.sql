SELECT * FROM Ai_Analytics.`ai job market dataset`;


select * from AI_Analysis;
select count( * ) from AI_Analysis;


-- 1.   Most Demanded jobs
SELECT job_title, COUNT(*) AS total_postings,
SUM(job_openings) AS total_openings
FROM AI_Analysis
GROUP BY job_title
ORDER BY total_openings DESC;

-- 2. Average salay by role

select job_title, ROUND(AVG(salary), 2) AS avg_salary
FROM AI_Analysis
GROUP BY job_title
ORDER BY avg_salary DESC;

-- 3. salary by experience 
SELECT experience_level, ROUND(AVG(salary), 2) AS avg_salary
FROM AI_Analysis
GROUP BY experience_level
ORDER BY avg_salary;
-- 4. remote vs hybrid vs onsite
SELECT remote_type, COUNT(*) AS postings, SUM(job_openings) AS openings,
ROUND(AVG(salary), 2) AS avg_salary
FROM AI_Analysis
GROUP BY remote_type;

-- 5. Hiring urgency
SELECT hiring_urgency, COUNT(*) AS total_jobs
FROM AI_Analysis
GROUP BY hiring_urgency
ORDER BY total_jobs DESC; 


