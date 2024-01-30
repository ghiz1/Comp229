/* 
Ghizlane Ez-Zarrad
301412844
2024-01-28
*/

import React from 'react';
import BBG from '../src/assets/BBG.jpg';
import { Link } from 'react-router-dom';
import '../src/index.css';
import '../src/home.css'; // or '../src/about.css'

export default function Home() {
  return (
    <section id="intro">
      <div className="container">
        <div className="imageContainer">
          <img src={BBG} alt="profile" className="image" />
        </div>
        <div className="textContainer">
          <span className="hello"><strong>Welcome to my website!</strong></span><br/>
          <span className="introText">Hello, I am a web developer</span> 
          <br/>
          <span className="introName"><stong> GHIZLANE EZ-ZARRAD</stong></span><br/>
          <p className="introPara">A curiosity-driven coder with a passion for designing and building user-centric, inclusive experiences on the web.</p>
          <Link to="/contact" className="btn">Hire me</Link>
          <Link to="/about" className="btn">More about me</Link>
        </div>
      </div>
    </section>
  );
}
