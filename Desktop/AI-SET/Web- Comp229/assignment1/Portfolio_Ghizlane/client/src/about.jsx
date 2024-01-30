/* 
Ghizlane Ez-Zarrad
301412844
2024-01-28
*/
import React from 'react';
import BBG from '../src/assets/BBG.jpg';
import resumePDF from '../src/assets/resume.pdf'; // Import your PDF file

import { Link } from 'react-router-dom';
import '../src/index.css';
import '../src/about.css';

export default function About() {
  return (
    <section id="skill">
      <div className="aboutContainer">
        <div>
          <img src={BBG} alt="profile" className="aboutImage" />
        </div>
        <span className="skillTitle"></span>
        <div className="aboutText">
          <span className="aboutText">
            My name is Ghizlane, and I'm a person who gets a rush from crafting websites that not only look stunning but also function cohessively. From the moment I started building codes, I have  always noticed something irrelivant about bringing ideas and implementing them on the web.  I'm still learning the magic of HTML, CSS, and JavaScript, and I'm stuck in this journey because I wanr to have the chance to prove my ability to shoew my creativity.  I am using technology to get inspired, to create connections and find new opprtunities. So, here I am, a beginner web developer with big dreams and passion, ready to take on challenges come my way and make my mark in this work. 
          </span>
          <br />
          <a href={resumePDF} className="download-btn" target="_blank" rel="noopener noreferrer">Download Resume</a>
          <br />
        </div>
      </div>
    </section>
  );
}
