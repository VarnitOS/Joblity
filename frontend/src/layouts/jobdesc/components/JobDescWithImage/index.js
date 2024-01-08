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

// @mui material components
import Card from "@mui/material/Card";
import Grid from "@mui/material/Grid";
import Icon from "@mui/material/Icon";

// Soft UI Dashboard React components
import SuiBox from "components/SuiBox";
import SuiTypography from "components/SuiTypography";

// Images
import wavesWhite from "assets/images/shapes/waves-white.svg";
import rocketWhite from "assets/images/illustrations/rocket-white.png";
import googleimg from "assets/images/logos/Google__G__Logo.svg.png";

import { useState, useEffect, React, useContext } from 'react'

import axios from 'axios'
import SuiButton from "../../../../components/SuiButton";


import AppContext from '../../../profileCreation/AppContext'



function JobDescWithImage() {
  const myContext = useContext(AppContext);
  const [companyData, setCompanyData] = useState("")

  useEffect(() => { 

    axios.post(`http://127.0.0.1:5000/jobs/info`, {
      headers: {
        "Access-Control-Allow-Origin": "*",
      
      },
      data: {
        'company': myContext.company,
        'job_title': myContext.title
      }
    })
      .then(response => {
       setCompanyData(response.data)

    
    });


    
  }, []);

  return (
    <Card>
      <SuiBox p={2}>

        <Grid container spacing={3}>
          <Grid item xs={12} lg={12}>
            <SuiBox display="flex" flexDirection="column" height="100%">
              <SuiBox pt={1} mb={0.5}>
                <SuiTypography variant="body2" color="text" fontWeight="medium">
                  {companyData.company}
                </SuiTypography>
              </SuiBox>
              <SuiTypography variant="h5" fontWeight="bold" gutterBottom>
              {companyData.job_title}
              </SuiTypography>
              <SuiBox mb={6} >
                <SuiTypography variant="body2" color="text">
                  {companyData.desc}
                </SuiTypography>
                <br />
                <SuiTypography variant="body2" color="text">Qualifications</SuiTypography>
                <ul style={{ 'padding-left': '20px' }}>
                   {companyData ? 
                       companyData.qualifications.map((item) => <li><SuiTypography variant="body2" color="text">
                       {item}
                       </SuiTypography></li>) : "No qualifications"
                  }
                </ul>
                <br />
                <SuiTypography variant="body2" fontWeight='bold' color="text">Disability & Accessibility Requirements:</SuiTypography>
                <ul style={{ 'padding-left': '20px', columnCount: 2 }}>
                  {companyData ? 
                       companyData.disabilities.map((item) => <li>
                         <SuiTypography variant="body2" color="text">
                          {item}
                          </SuiTypography></li>) : "No disability requirements"
                  }
                </ul>
                <div>
                  <br/>
                </div>
                <SuiButton variant="gradient" color="dark" >
                  &nbsp;Apply now!
                </SuiButton>

              </SuiBox>
            </SuiBox>
          </Grid>

        </Grid>
      </SuiBox>
    </Card>
  );
}

export default JobDescWithImage;
