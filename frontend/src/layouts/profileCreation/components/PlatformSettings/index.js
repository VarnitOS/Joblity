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

import { useState } from "react";

// @mui material components
import Card from "@mui/material/Card";
import Switch from "@mui/material/Switch";

// Soft UI Dashboard PRO React components
import SuiBox from "components/SuiBox";
import SuiTypography from "components/SuiTypography";


function PlatformSettings() {
  const [visualImpaired, setVisualImpaired] = useState(true);
  const [hearingImpaired, setHearingImpaired] = useState(false);
  const [wheelChairImpaired, setWheelChairImpaired] = useState(true);
  const [autism, setAutism] = useState(true);
  const [workHands, setWorkHands] = useState(true);
  const [talkingToStrangers, setTalkingToStrangers] = useState(false);
  const [workWithComputers, setWorkWithComputers] = useState(true);
  const [creativty, setCreativity] = useState(true);

  return (
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
            <Switch checked={creativty} onChange={() => setCreativity(!creativty)} />
          </SuiBox>
          <SuiBox width="80%" ml={2}>
            <SuiTypography variant="button" fontWeight="regular" color="text">
            Do you face difficulties using your creativity to make something new?
            </SuiTypography>
          </SuiBox>
        </SuiBox>
      </SuiBox>
    </Card>
  );
}

export default PlatformSettings;
