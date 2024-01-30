/* 
Ghizlane Ez-Zarrad
301412844
2024-01-28
*/
import React from 'react';
import { Link } from 'react-router-dom';
import project1 from '../src/assets/project1.jpg';
import project2 from '../src/assets/project2.png';
import project3 from '../src/assets/project3.jpg';
import project4 from '../src/assets/project4.jpg';
import '../src/project.css';

const Project = () => {
  return (
    <section id="works">
      <h2 className="worksTitle">My Projects</h2>
      <p className="worksDesc">
      In my past projects, I've tackled different challenges, and each one helped me grow as a web developer. One project I'm proud of is creating a system to predict fog at airports, where I used data analysis to make things safer. I also led a team to make a digital tool for a hospital, making it easier to do administrative work and take care of patients. Another project I did was making a tool to figure out how different parts of a small city depend on each other, which helps plan things better. I've also taken part in hackathons, working with small businesses to come up with new ideas and get better at coding in real life.      </p>
      <div className="worksImgs">
        <img src={project1} alt="Project 1" className="worksImg" />
        <img src={project2} alt="Project 2" className="worksImg" />
        <img src={project3} alt="Project 3" className="worksImg" />
        <img src={project4} alt="Project 4" className="worksImg" />
      </div>
      <Link to="/projects" className="workBtn">See More</Link>
    </section>
  );
};

export default Project;