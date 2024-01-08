// @mui material components
import Card from "@mui/material/Card";
import Grid from "@mui/material/Grid";
import Icon from "@mui/material/Icon";

// Soft UI Dashboard React components
import SuiBox from "components/SuiBox";
import SuiTypography from "components/SuiTypography";
import SuiButton from "components/SuiButton"

import { Link, useHistory } from "react-router-dom";


import { useState, useEffect, React, useContext } from 'react'


import bankTeller from "assets/images/jobs/bank-teller.png";
import callCenterService from "assets/images/jobs/call-center-service.png";
import cashier from "assets/images/jobs/cashier.png";
import clerk from "assets/images/jobs/clerk.png";
import copywriter from "assets/images/jobs/copywriter.png";
import dataJobs from "assets/images/jobs/data-jobs.png";
import factoryWorker from "assets/images/jobs/factory-worker.png";
import flyerDistributer from "assets/images/jobs/flyer-distributer.png";
import foodProcessingWorker from "assets/images/jobs/food-processing-worker.png";
import helpdesk from "assets/images/jobs/helpdesk.png";
import humanResources from "assets/images/jobs/human-resources.png";
import mailman from "assets/images/jobs/mailman.png";
import manufacturingWorker from "assets/images/jobs/manufacturing-worker.png";
import marketing from "assets/images/jobs/marketing.png";
import packingWorker from "assets/images/jobs/packing-worker.png";
import salesman from "assets/images/jobs/salesman.png";
import truckDriver from "assets/images/jobs/truck-driver.png";
import waiter from "assets/images/jobs/waiter.png";
import webDeveloper from "assets/images/jobs/web-developer.png";


import AppContext from '../../../profileCreation/AppContext'
import './index.css'


function getLogo(title) {
    const logoMapping = new Map();
    logoMapping['Flyer Distributer'] = flyerDistributer;
    logoMapping.Mailman = mailman;
    logoMapping['Delivery Staff'] = truckDriver;
    logoMapping['Delivery Personnel'] = truckDriver;
    logoMapping['Truck Driver'] = truckDriver;
    logoMapping.Cashier = cashier;
    logoMapping.Waiter = waiter;
    logoMapping['Call Center Worker'] = callCenterService;
    logoMapping['Customer Service'] = callCenterService;
    logoMapping['Bank Teller'] = bankTeller;
    logoMapping['Frontline Sales'] = salesman;
    logoMapping['Factory/General Worker'] = factoryWorker;
    logoMapping['Food Processing Worker'] = foodProcessingWorker;
    logoMapping['Packing Worker'] = packingWorker;
    logoMapping['Manufacturing Worker'] = manufacturingWorker;
    logoMapping['Data Entry'] = dataJobs;
    logoMapping['Data Processing Clerk'] = dataJobs;
    logoMapping['Data Analyst'] = dataJobs;
    logoMapping['Business Analyst'] = dataJobs;
    logoMapping['Data Engineer'] = dataJobs;
    logoMapping['IT Help Desk'] = helpdesk;
    logoMapping['Web Developer'] = webDeveloper;
    logoMapping['Advertising Copywriter'] = copywriter;
    logoMapping['Marketing Copywriter'] = copywriter;
    logoMapping['Advertising Staff'] = marketing;
    logoMapping['Marketing Staff'] = marketing;
    logoMapping['Admin Assistant'] = clerk;
    logoMapping['HR Staff'] = humanResources;
    logoMapping['Research Assistant'] = clerk;
    logoMapping['Personal Assistant'] = clerk;
    return logoMapping[title];
}

function RecoBox({key, company, title, desc, isTop}) {
  const myContext = useContext(AppContext);

  const history = useHistory();

  const saveStateAndRedirect = () => {
    myContext.setCompanyValue(company)
    myContext.setTitleValue(title)
    const path = '/job-description'
    history.push(path)
  }

  const logo = getLogo(title);

  return (
    <div>
    <Card className = "onHover">
      <SuiBox p={2}>
        <Grid container spacing={3}>
          <Grid item xs={12} lg={12}>
            <SuiBox  height="100%">
            {/* <SuiBox height="100%"> */}
              <SuiBox>
                <Grid container alignItems="center">
                  <Grid item xs = {isTop? 2:3}>
                    <SuiBox ml={2} lineHeight={1}
                      variant="gradient"
                      bgColor="info"
                      color="white"
                      width="5rem"
                      height="5rem"
                      borderRadius="md"
                      display="flex"
                      justifyContent="center"
                      alignItems="center"
                      shadow="md"
                      sx={{
                          backgroundImage: ({ functions: { rgba, linearGradient }, palette: { gradients } }) =>
                            `${linearGradient(
                              rgba(gradients.light.main, 0.1),
                              rgba(gradients.light.state, 0.1)
                            )}, url(${logo})`,
                          backgroundSize: "contain",
                          backgroundPosition: "50%",
                          backgroundRepeat: "no-repeat"
                        }}>
                        <></>
                    </SuiBox>
                  </Grid>
                  
                  <Grid item xs={8}>
                    <SuiBox ml={1} lineHeight={1}>
                      <SuiBox pt={1} mb={0.5}>
                        <SuiTypography variant="body2" color="text" fontWeight="medium">
                          {company}
                        </SuiTypography>
                      </SuiBox>
                      <SuiTypography variant="h5" fontWeight="bold" gutterBottom>
                        {title}
                      </SuiTypography>
                    </SuiBox>
                  </Grid>
                </Grid>
              </SuiBox>

              <SuiBox mb={1} mt={2}>
                <SuiTypography variant="body2" color="text">
                  {desc}
                </SuiTypography>
              </SuiBox>
              <SuiButton variant="gradient" color="dark" onClick={saveStateAndRedirect} className = "float_right">
                &nbsp;Read more
              </SuiButton>
            </SuiBox>
          </Grid>
        </Grid>
      </SuiBox>
    </Card>
    </div>
  );
}

export default RecoBox;