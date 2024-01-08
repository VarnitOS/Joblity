/**
=========================================================
* Soft UI Dashboard React - v2.0.0
=========================================================

* Product Page: https://www.creative-tim.com/product/soft-ui-dashboard-pro-material-ui
* Copyright 2021 Creative Tim (https://www.creative-tim.com)

Coded by www.creative-tim.com

 =========================================================

* The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

// @mui material components
import Grid from "@mui/material/Grid";
import Card from "@mui/material/Card";
import { Link, useHistory } from "react-router-dom";


// @mui icons
import FacebookIcon from "@mui/icons-material/Facebook";
import TwitterIcon from "@mui/icons-material/Twitter";
import InstagramIcon from "@mui/icons-material/Instagram";

// Soft UI Dashboard React components
import SuiBox from "components/SuiBox";
import SuiTypography from "components/SuiTypography";

// Soft UI Dashboard React example components
import DashboardLayout from "examples/LayoutContainers/DashboardLayout";
import Footer from "examples/Footer";
import CreateProfileCard from "examples/Cards/InfoCards/CreateProfileCard";
import ProfilesList from "examples/Lists/ProfilesList";
// import DefaultProjectCard from "examples/Cards/ProjectCards/DefaultProjectCard";
// import PlaceholderCard from "examples/Cards/PlaceholderCard";

// Overview page components
import Header from "layouts/profile/components/Header";
import PlatformSettings from "layouts/profileCreation/components/PlatformSettings";

// Data
import profilesListData from "layouts/profileCreation/data/profilesListData";

import { FormControl, InputLabel, Input, FormHelperText } from '@mui/material';
import SuiButton from "components/SuiButton";
import Icon from "@mui/material/Icon";
import Switch from "@mui/material/Switch";

import "./index.css"

import { useState, useEffect, React, useContext } from 'react'

import axios from 'axios';

import AppContext from './AppContext';

function Overview() {
  const history = useHistory();

  const [nameSetting, setName] = useState("")
  const [mobileSetting, setMobile] = useState("")
  const [emailSetting, setEmail] = useState("")
  const [locationSetting, setLocation] = useState("")
  const [descriptionSetting, setDescription] = useState("")
  const myContext = useContext(AppContext);

  const [visualImpaired, setVisualImpaired] = useState(false);
  const [hearingImpaired, setHearingImpaired] = useState(false);
  const [wheelChairImpaired, setWheelChairImpaired] = useState(false);
  const [autism, setAutism] = useState(false);
  const [workHands, setWorkHands] = useState(false);
  const [talkingToStrangers, setTalkingToStrangers] = useState(false);
  const [workWithComputers, setWorkWithComputers] = useState(false);
  const [creativity, setCreativity] = useState(false);

  const [initialQuestionList, setInitialQuestionList] = useState([])

  const redirectToRecommender = () => {
    
    
    const inputToQuestions = [visualImpaired, hearingImpaired, wheelChairImpaired, autism, workHands, talkingToStrangers, workWithComputers, creativity]
    for (let i = 0; i < inputToQuestions.length; i += 1) {
      if (inputToQuestions[i]) {
        inputToQuestions[i] = 1
      } else {
        inputToQuestions[i] = 0
      }
    }

    myContext.setQuestionValue(inputToQuestions)
    const path = '/recommender'
    history.push(path)
  }


  useEffect(() => { 
    
    setName(myContext.name)
    setMobile(myContext.mobile)
    setEmail(myContext.email)
    setLocation(myContext.location)
    setDescription(myContext.description)
    setInitialQuestionList(myContext.questionList)

    if (myContext.questionList[0] === 1) {
      setVisualImpaired(true);
    } 

    if (myContext.questionList[1] === 1) {
      setHearingImpaired(true);
    } 

    if (myContext.questionList[2] === 1) {
      setWheelChairImpaired(true);
    } 

    if (myContext.questionList[3] === 1) {
      setAutism(true);
    } 

    if (myContext.questionList[4] === 1) {
      setWorkHands(true);
    } 

    if (myContext.questionList[5] === 1) {
      setTalkingToStrangers(true);
    } 

    if (myContext.questionList[6] === 1) {
      setWorkWithComputers(true);
    } 

    if (myContext.questionList[7] === 1) {
      setCreativity(true);
    } 


  }, []);

  return (
    <DashboardLayout>
      <Header />
      
      <SuiBox mt={12} mb={12}>
        <Grid container spacing={3}>

          <Grid item xs={12} md={12} xl={6}>
            <CreateProfileCard
              title="profile information"
              description= {descriptionSetting}
              info={{
                fullName: nameSetting,
                mobile: mobileSetting,
                email: emailSetting,
                location: locationSetting,
              }}
              social={[
                {
                  link: "https://www.facebook.com/CreativeTim/",
                  icon: <FacebookIcon />,
                  color: "facebook",
                },
                {
                  link: "https://twitter.com/creativetim",
                  icon: <TwitterIcon />,
                  color: "twitter",
                },
                {
                  link: "https://www.instagram.com/creativetimofficial/",
                  icon: <InstagramIcon />,
                  color: "instagram",
                },
              ]}
              action={{ route: "", tooltip: "Edit Profile" }}
            />
          </Grid>
          <Grid item xs={12} md={6} xl={6}>
          <Card>
      <SuiBox pt={2} px={2}>
        <SuiTypography variant="h6" fontWeight="medium" textTransform="capitalize">
          Additional questions to understand you better
        </SuiTypography>
      </SuiBox>
      <SuiBox pt={1.5} pb={2} px={2} lineHeight={1.25}>
        <SuiTypography variant="caption" fontWeight="bold" color="text" textTransform="uppercase">
          Disability questions
        </SuiTypography>
        <SuiBox display="flex" py={1} mb={0.25}>
          <SuiBox mt={0.25}>
            <Switch checked={visualImpaired} onChange={() => setVisualImpaired(!visualImpaired)} />
          </SuiBox>
          <SuiBox width="80%" ml={2}>
            <SuiTypography variant="button" fontWeight="regular" color="text">
             Are you visually impaired?
            </SuiTypography>
          </SuiBox>
        </SuiBox>
        <SuiBox display="flex" py={1} mb={0.25}>
          <SuiBox mt={0.25}>
            <Switch checked={hearingImpaired} onChange={() => setHearingImpaired(!hearingImpaired)} />
          </SuiBox>
          <SuiBox width="80%" ml={2}>
            <SuiTypography variant="button" fontWeight="regular" color="text">
             Do you have hearing problems?
            </SuiTypography>
          </SuiBox>
        </SuiBox>
        <SuiBox display="flex" py={1} mb={0.25}>
          <SuiBox mt={0.25}>
            <Switch checked={wheelChairImpaired} onChange={() => setWheelChairImpaired(!wheelChairImpaired)} />
          </SuiBox>
          <SuiBox width="80%" ml={2}>
            <SuiTypography variant="button" fontWeight="regular" color="text">
              Do you require wheelchairs to move around?
            </SuiTypography>
          </SuiBox>
        </SuiBox>
        <SuiBox display="flex" py={1} mb={0.25}>
          <SuiBox mt={0.25}>
            <Switch checked={autism} onChange={() => setAutism(!autism)} />
          </SuiBox>
          <SuiBox width="80%" ml={2}>
            <SuiTypography variant="button" fontWeight="regular" color="text">
              Are you autistic?
            </SuiTypography>
          </SuiBox>
        </SuiBox>
        <SuiBox display="flex" py={1} mb={0.25}>
          <SuiBox mt={0.25}>
            <Switch checked={workHands} onChange={() => setWorkHands(!workHands)} />
          </SuiBox>
          <SuiBox width="80%" ml={2}>
            <SuiTypography variant="button" fontWeight="regular" color="text">
            Do you face difficulties working with your hands?
            </SuiTypography>
          </SuiBox>
        </SuiBox>
        <SuiBox mt={3}>
          <SuiTypography variant="caption" fontWeight="bold" color="text" textTransform="uppercase">
            Preferences
          </SuiTypography>
        </SuiBox>
        <SuiBox display="flex" py={1} mb={0.25}>
          <SuiBox mt={0.25}>
            <Switch checked={talkingToStrangers} onChange={() => setTalkingToStrangers(!talkingToStrangers)} />
          </SuiBox>
          <SuiBox width="80%" ml={2}>
            <SuiTypography variant="button" fontWeight="regular" color="text">
            Do you face difficulties talking to strangers?
            </SuiTypography>
          </SuiBox>
        </SuiBox>
        <SuiBox display="flex" py={1} mb={0.25}>
          <SuiBox mt={0.25}>
            <Switch checked={workWithComputers} onChange={() => setWorkWithComputers(!workWithComputers)} />
          </SuiBox>
          <SuiBox width="80%" ml={2}>
            <SuiTypography variant="button" fontWeight="regular" color="text">
            Do you face difficulties working with computers?
            </SuiTypography>
          </SuiBox>
        </SuiBox>
        <SuiBox display="flex" py={1} mb={0.25}>
          <SuiBox mt={0.25}>
            <Switch checked={creativity} onChange={() => setCreativity(!creativity)} />
          </SuiBox>
          <SuiBox width="80%" ml={2}>
            <SuiTypography variant="button" fontWeight="regular" color="text">
            Do you face difficulties using your creativity to make something new?
            </SuiTypography>
          </SuiBox>
        </SuiBox>
      </SuiBox>
    </Card>
          </Grid>
          {/* <Grid item xs={12} xl={4}>
            <ProfilesList title="conversations" profiles={profilesListData} />
          </Grid> */}
        </Grid>
        <div className="text-center padding-top">
        <SuiButton variant="gradient" color="dark" onClick={redirectToRecommender} >
          &nbsp;Recommend me!
        </SuiButton>
        </div>
      </SuiBox>
      <Footer />
    </DashboardLayout>
  );
}

export default Overview;
