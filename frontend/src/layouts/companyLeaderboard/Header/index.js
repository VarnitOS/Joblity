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

import { useState, useEffect } from "react";

// @mui material components
import Card from "@mui/material/Card";
import Grid from "@mui/material/Grid";
import AppBar from "@mui/material/AppBar";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";

// Soft UI Dashboard PRO React components
import SuiBox from "components/SuiBox";
import SuiTypography from "components/SuiTypography";
import SuiAvatar from "components/SuiAvatar";

// Soft UI Dashboard PRO React example components
import DashboardNavbar from "examples/Navbars/DashboardNavbar";

// Soft UI Dashboard PRO React icons
import Cube from "examples/Icons/Cube";
import Document from "examples/Icons/Document";
import Settings from "examples/Icons/Settings";

// Soft UI Dashboard PRO React base styles
import breakpoints from "assets/theme/base/breakpoints";

// Images
import burceMars from "assets/images/bruce-mars.jpg";
import leaderboardHeader from "assets/images/leaderboard-header-final.jpg";

function Header() {
  const [tabsOrientation, setTabsOrientation] = useState("horizontal");
  const [tabValue, setTabValue] = useState(0);

  useEffect(() => {
    // A function that sets the orientation state of the tabs.
    function handleTabsOrientation() {
      return window.innerWidth < breakpoints.values.sm
        ? setTabsOrientation("vertical")
        : setTabsOrientation("horizontal");
    }

    /** 
     The event listener that's calling the handleTabsOrientation function when resizing the window.
    */
    window.addEventListener("resize", handleTabsOrientation);

    // Call the handleTabsOrientation function to set the state with the initial value.
    handleTabsOrientation();

    // Remove event listener on cleanup
    return () => window.removeEventListener("resize", handleTabsOrientation);
  }, [tabsOrientation]);

  const handleSetTabValue = (event, newValue) => setTabValue(newValue);

  return (
    // <SuiBox position="relative">
    <>
      <DashboardNavbar absolute dark />
      <SuiBox
        display="flex"
        alignItems="center"
        position="relative"
        minHeight="30.75rem"
        borderRadius="xl"
        mb={0}
        sx={{
          backgroundImage: ({ functions: { rgba, linearGradient }, palette: { gradients } }) =>
            `${linearGradient(
              rgba(gradients.light.main, 0.1),
              rgba(gradients.light.state, 0.1)
            )}, url(${leaderboardHeader})`,
          backgroundSize: "contain",
          backgroundPosition: "50%",
          backgroundRepeat: "no-repeat"
        }}
      />
    </>
  );
}

export default Header;