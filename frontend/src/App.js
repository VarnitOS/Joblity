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

import { useState, useEffect, useMemo } from "react";

// react-router components
import { Route, Switch, Redirect, useLocation } from "react-router-dom";

// @mui material components
import { ThemeProvider } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import Icon from "@mui/material/Icon";

// Soft UI Dashboard PRO React components
import SuiBox from "components/SuiBox";

// Soft UI Dashboard PRO React example components
import Sidenav from "examples/Sidenav";
import Configurator from "examples/Configurator";

// Soft UI Dashboard PRO React themes
import theme from "assets/theme";
import themeRTL from "assets/theme/theme-rtl";

// RTL plugins
import rtlPlugin from "stylis-plugin-rtl";
import { CacheProvider } from "@emotion/react";
import createCache from "@emotion/cache";

// Soft UI Dashboard PRO React routes
import routes from "routes";

// Soft UI Dashboard PRO React contexts
import { useSoftUIController, setMiniSidenav, setOpenConfigurator } from "context";

// Images
import brand from "assets/images/logo-jobility.png";

import AppContext from "layouts/profileCreation/AppContext.js";

import Dashboard from "layouts/dashboard";
import Tables from "layouts/tables";

import JobDesc from "layouts/jobdesc";



export default function App() {
  const [controller, dispatch] = useSoftUIController();
  const { miniSidenav, direction, layout, openConfigurator, sidenavColor } = controller;
  const [onMouseEnter, setOnMouseEnter] = useState(false);
  const [rtlCache, setRtlCache] = useState(null);
  const { pathname } = useLocation();

  // Cache for the rtl
  useMemo(() => {
    const cacheRtl = createCache({
      key: "rtl",
      stylisPlugins: [rtlPlugin],
    });

    setRtlCache(cacheRtl);
  }, []);

  // Open sidenav when mouse enter on mini sidenav
  const handleOnMouseEnter = () => {
    if (miniSidenav && !onMouseEnter) {
      setMiniSidenav(dispatch, false);
      setOnMouseEnter(true);
    }
  };

  // Close sidenav when mouse leave mini sidenav
  const handleOnMouseLeave = () => {
    if (onMouseEnter) {
      setMiniSidenav(dispatch, true);
      setOnMouseEnter(false);
    }
  };

  // Change the openConfigurator state
  const handleConfiguratorOpen = () => setOpenConfigurator(dispatch, !openConfigurator);

  // Setting the dir attribute for the body element
  useEffect(() => {
    document.body.setAttribute("dir", direction);
  }, [direction]);

  // Setting page scroll to 0 when changing the route
  useEffect(() => {
    document.documentElement.scrollTop = 0;
    document.scrollingElement.scrollTop = 0;
  }, [pathname]);

  const [nameValue, setNameValue] = useState("Mr Adam")
  const [emailValue, setEmailValue] = useState("adam@mails.com")
  const [mobileValue, setMobileValue] = useState("+65 1234 5678")
  const [locationValue, setLocationValue] = useState("Bukit Timah")
  const [descriptionValue, setDescriptionValue] = useState("ADD A PROFILE DESCRIPTION")
  const [questionValue, setQuestionValue] = useState([0, 0, 0, 0, 0, 0, 0, 0])
  const [companyValue, setCompanyValue] = useState("")
  const [titleValue, setTitleValue] = useState("")

  const userSettings = {
    name: nameValue,
    email: emailValue,
    mobile: mobileValue,
    location: locationValue,
    description: descriptionValue,
    questionList: questionValue,
    company: companyValue,
    title: titleValue,
    setNameValue,
    setEmailValue,
    setMobileValue,
    setLocationValue,
    setDescriptionValue,
    setQuestionValue,
    setCompanyValue,
    setTitleValue
  };

  const getRoutes = (allRoutes) =>
    allRoutes.map((route) => {
      if (route.collapse) {
        return getRoutes(route.collapse);
      }

      if (route.route) {
        // console.log(userSettings)
        return (
          <div>

            <Route exact path={route.route} component={route.component} key={route.key} />
            {/* <Route path = "/jobdesc" component = {JobDesc}/> */}
          </div>
        )
      }

      return null;
    });

  const configsButton = (
    <SuiBox
      display="flex"
      justifyContent="center"
      alignItems="center"
      width="3.5rem"
      height="3.5rem"
      bgColor="white"
      shadow="sm"
      borderRadius="50%"
      position="fixed"
      right="2rem"
      bottom="2rem"
      zIndex={99}
      color="dark"
      sx={{ cursor: "pointer" }}
      onClick={handleConfiguratorOpen}
    >
      <Icon fontSize="default" color="inherit">
        settings
      </Icon>
    </SuiBox>
  );

  return direction === "rtl" ? (
    <CacheProvider value={rtlCache}>
      <ThemeProvider theme={themeRTL}>
        <CssBaseline />
        {layout === "dashboard" && (
          <>
            <Sidenav
              color={sidenavColor}
              brand={brand}
              brandName="Jobility"
              routes={routes}
              onMouseEnter={handleOnMouseEnter}
              onMouseLeave={handleOnMouseLeave}
            />
            {/* <Configurator />
            {configsButton} */}
          </>
        )}
        {layout === "vr" && <Configurator />}
        <Switch>
          <AppContext.Provider value={userSettings}>
            {getRoutes(routes)}


            <Redirect from="*" to="/profile" />
          </AppContext.Provider>
        </Switch>
      </ThemeProvider>
    </CacheProvider>
  ) : (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {layout === "dashboard" && (
        <>
          <Sidenav
            color={sidenavColor}
            brand={brand}
            brandName="Jobility"
            routes={routes}
            onMouseEnter={handleOnMouseEnter}
            onMouseLeave={handleOnMouseLeave}
          />
          {/* <Configurator />
          {configsButton} */}
        </>
      )}
      {layout === "vr" && <Configurator />}
      <Switch>
        <AppContext.Provider value={userSettings}>
          {getRoutes(routes)}
          <Redirect from="*" to="/profile" />
        </AppContext.Provider>
      </Switch>
    </ThemeProvider>
  );
}
