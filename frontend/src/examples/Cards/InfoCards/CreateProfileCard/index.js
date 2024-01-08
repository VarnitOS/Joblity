/**
=========================================================
* Soft UI Dashboard React - v3.0.0
=========================================================

* Product Page: https://www.creative-tim.com/product/soft-ui-dashboard-react
* Copyright 2021 Creative Tim (https://www.creative-tim.com)

Coded by www.creative-tim.com

 =========================================================

* The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

// react-routers components
import { Link, useHistory } from "react-router-dom";

// prop-types is library for typechecking of props
import PropTypes from "prop-types";

// @mui material components
import Card from "@mui/material/Card";
import Divider from "@mui/material/Divider";
import Tooltip from "@mui/material/Tooltip";
import Icon from "@mui/material/Icon";

// Soft UI Dashboard PRO React components
import SuiBox from "components/SuiBox";
import SuiTypography from "components/SuiTypography";

// Soft UI Dashboard PRO React base styles
import colors from "assets/theme/base/colors";
import typography from "assets/theme/base/typography";

import SuiButton from "components/SuiButton";

import Box from "@mui/material/Box";
import Input from "@mui/material/Input";
import InputLabel from "@mui/material/InputLabel";
import InputAdornment from "@mui/material/InputAdornment";
import FormControl from "@mui/material/FormControl";
import TextField from "@mui/material/TextField";
import AccountCircle from "@mui/icons-material/AccountCircle";

import "./index.css";

import SpeechRecognition, { useSpeechRecognition } from "react-speech-recognition";

import { useState, useEffect, React, useContext } from "react";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";

import AppContext from "../../../../layouts/profileCreation/AppContext.js";

function CreateProfileCard({ title, description, info, social, action }) {
  const labels = [];
  const values = [];
  const { socialMediaColors } = colors;
  const { size } = typography;

  const myContext = useContext(AppContext);

  const [nameSetting, setName] = useState("");
  const [mobileSetting, setMobile] = useState("");
  const [emailSetting, setEmail] = useState("");
  const [locationSetting, setLocation] = useState("");
  const [transcriptResult, setTranscriptResult] = useState("");

  const history = useHistory();

  const [text, setText] = useState("");
  const [listeningBoolean, setListeningBoolean] = useState(false);

  const language = ["en-US", "zh-CN", "ko", "ms-MY"];
  const languageFull = ["English", "Chinese", "Korea", "Malay"];
  const [languageChosen, setLanguageChosen] = useState("en-US");
  const [languageChosenFull, setLanguageChosenFull] = useState(0);

  const { transcript, listening, resetTranscript, browserSupportsSpeechRecognition } =
    useSpeechRecognition();

  useEffect(() => {
    setName(myContext.name);
    setMobile(myContext.mobile);
    setEmail(myContext.email);
    setLocation(myContext.location);
  }, []);

  if (!browserSupportsSpeechRecognition) {
    return <span>Browser doesn't support speech recognition.</span>;
  }

  const changeLanguage = (event) => {
    setLanguageChosen(language[event.target.value]);
    setLanguageChosenFull(event.target.value);
  };

  const startListening = () => {
    setListeningBoolean(true);
    resetTranscript();
    SpeechRecognition.startListening({ continuous: true, language: languageChosen });
  };

  const stopListening = () => {
    SpeechRecognition.stopListening();
    setListeningBoolean(false);
    setText(transcript);
  };

  const [editProfile, setEditProfile] = useState(false);

  // Convert this form `objectKey` of the object key in to this `object key`
  Object.keys(info).forEach((el) => {
    if (el.match(/[A-Z\s]+/)) {
      const uppercaseLetter = Array.from(el).find((i) => i.match(/[A-Z]+/));
      const newElement = el.replace(uppercaseLetter, ` ${uppercaseLetter.toLowerCase()}`);

      labels.push(newElement);
    } else {
      labels.push(el);
    }
  });

  // Push the object values into the values array
  Object.values(info).forEach((el) => values.push(el));

  // Render the card info items
  const renderItems = labels.map((label, key) => (
    <SuiBox key={label} display="flex" py={1} pr={2}>
      <SuiTypography variant="button" fontWeight="bold" textTransform="capitalize">
        {label}: &nbsp;
      </SuiTypography>
      <SuiTypography variant="button" fontWeight="regular" color="text">
        &nbsp;{values[key]}
      </SuiTypography>
    </SuiBox>
  ));

  const editOwnProfile = () => {
    if (editProfile) {
      stopListening();
      setTranscriptResult(transcript);

      myContext.setDescriptionValue(text);
      myContext.setNameValue(nameSetting);
      myContext.setMobileValue(mobileSetting);
      myContext.setEmailValue(emailSetting);
      myContext.setLocationValue(locationSetting);

      setEditProfile(!editProfile);

      const path = "/dashboard";
      history.push(path);
      return;
    }

    setEditProfile(!editProfile);
  };

  const handleWordChange = (event) => {
    setText(event.target.value);
  };

  // Render the card social media icons
  const renderSocial = social.map(({ link, icon, color }) => (
    <SuiBox
      key={color}
      component="a"
      href={link}
      target="_blank"
      rel="noreferrer"
      fontSize={size.lg}
      color={socialMediaColors[color].main}
      pr={1}
      pl={0.5}
      lineHeight={1}
    >
      {icon}
    </SuiBox>
  ));

  return (
    <div>
      {editProfile ? (
        <Card sx={{ height: "100%" }}>
          <SuiBox display="flex" justifyContent="space-between" alignItems="center" pt={2} px={2}>
            <SuiTypography variant="h6" fontWeight="medium" textTransform="capitalize">
              {title}
            </SuiTypography>
            <SuiTypography component={Link} to={action.route} variant="body2" color="secondary">
              {/* <Tooltip title={action.tooltip} placement="top">
            <Icon>edit</Icon>
          </Tooltip> */}
            </SuiTypography>
          </SuiBox>
          <SuiBox p={2}>
            <TextField
              id="input-with-icon-textfield"
              label="Name"
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <AccountCircle />
                  </InputAdornment>
                ),
              }}
              variant="standard"
              onChange={(e) => setName(e.target.value)}
              value={nameSetting}
            />
          </SuiBox>
          <SuiBox p={2}>
            <TextField
              id="input-with-icon-textfield"
              label="Mobile"
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <AccountCircle />
                  </InputAdornment>
                ),
              }}
              variant="standard"
              value={mobileSetting}
              onChange={(e) => setMobile(e.target.value)}
            />
          </SuiBox>
          <SuiBox p={2}>
            <TextField
              id="input-with-icon-textfield"
              label="Email"
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <AccountCircle />
                  </InputAdornment>
                ),
              }}
              variant="standard"
              value={emailSetting}
              onChange={(e) => setEmail(e.target.value)}
            />
          </SuiBox>
          <SuiBox p={2}>
            <TextField
              id="input-with-icon-textfield"
              label="Location"
              value={locationSetting}
              onChange={(e) => setLocation(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <AccountCircle />
                  </InputAdornment>
                ),
              }}
              variant="standard"
            />
          </SuiBox>

          <SuiBox p={2}>
            <SuiTypography variant="h5" fontWeight="medium">
              Record additional profile information here
            </SuiTypography>

            <SuiTypography variant="h5" fontWeight="medium">
              Microphone: {listening ? "on" : "off"}
            </SuiTypography>

            {!listeningBoolean ? (
              <form>
                Select language:
                <div />
                <select
                  name="languages"
                  id="languages"
                  onChange={changeLanguage}
                  value={languageChosenFull}
                >
                  <option value={0}>English</option>
                  <option value={1}>Chinese</option>
                  <option value={2}>Korea</option>
                  <option value={3}>Malay</option>
                </select>
              </form>
            ) : null}

            <SuiButton variant="gradient" color="dark" onClick={startListening}>
              &nbsp;Start recording
            </SuiButton>
            <SuiButton variant="gradient" color="dark" onClick={stopListening}>
              &nbsp;Stop
            </SuiButton>

            <div className="full-width">
              <TextField
                multiline
                rows={4}
                fullWidth
                value={listeningBoolean ? transcript : text}
                onChange={handleWordChange}
              />
            </div>
          </SuiBox>
          <div className="center">
            {listeningBoolean ? null : (
              <SuiButton variant="gradient" color="dark" onClick={editOwnProfile}>
                <Icon sx={{ fontWeight: "bold" }}>add</Icon>
                &nbsp;Save Profile
              </SuiButton>
            )}
          </div>
          
        </Card>
        
      ) : (
        <Card sx={{ height: "100%" }}>
          <SuiBox display="flex" justifyContent="space-between" alignItems="center" pt={2} px={2}>
            <SuiTypography variant="h6" fontWeight="medium" textTransform="capitalize">
              {title}
            </SuiTypography>
          </SuiBox>
          <SuiBox p={2}>
            <SuiBox mb={2} lineHeight={1}>
              <SuiTypography variant="button" color="text" fontWeight="regular">
                {description}
              </SuiTypography>
            </SuiBox>
            <SuiBox opacity={0.3}>
              <Divider />
            </SuiBox>
            <SuiBox>
              {renderItems}
            </SuiBox>
          </SuiBox>
          <div className="center">
            <SuiButton variant="gradient" color="dark" onClick={editOwnProfile}>
              <Icon sx={{ fontWeight: "bold" }}>add</Icon>
              &nbsp;Edit Profile
            </SuiButton>
          </div>
        </Card>
      )}
    </div>
  );
}

// Typechecking props for the ProfileInfoCard
CreateProfileCard.propTypes = {
  title: PropTypes.string.isRequired,
  description: PropTypes.string.isRequired,
  info: PropTypes.objectOf(PropTypes.string).isRequired,
  social: PropTypes.arrayOf(PropTypes.object).isRequired,
  action: PropTypes.shape({
    route: PropTypes.string.isRequired,
    tooltip: PropTypes.string.isRequired,
  }).isRequired,
};

export default CreateProfileCard;
