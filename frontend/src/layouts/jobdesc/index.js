// @mui material components
import Card from "@mui/material/Card";

// @mui icons
import FacebookIcon from "@mui/icons-material/Facebook";
import TwitterIcon from "@mui/icons-material/Twitter";
import InstagramIcon from "@mui/icons-material/Instagram";

// Soft UI Dashboard React components
import SuiTypography from "components/SuiTypography";

// Soft UI Dashboard React example components
import ProfileInfoCard from "examples/Cards/InfoCards/ProfileInfoCard";
import ProfilesList from "examples/Lists/ProfilesList";

// Overview page components
import PlatformSettings from "layouts/profile/components/PlatformSettings";

// Data
import profilesListData from "layouts/profile/data/profilesListData";

// Images


import JobDescWithImage from "layouts/jobdesc/components/JobDescWithImage";

import SuiButton from "components/SuiButton";


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
import Grid from "@mui/material/Grid";

// Soft UI Dashboard React components
import SuiBox from "components/SuiBox";

// Soft UI Dashboard React components

// Soft UI Dashboard React example components
import DashboardLayout from "examples/LayoutContainers/DashboardLayout";
import DashboardNavbar from "examples/Navbars/DashboardNavbar";
import Footer from "examples/Footer";

// Billing page components
import BillingInformation from "layouts/jobdesc/components/BillingInformation";
import Transactions from "layouts/billing/components/Transactions";

function JobDesc() {
  
  return (
    <DashboardLayout>
      <DashboardNavbar />
      <SuiBox mt={4}>
        <SuiBox mb={3}>
          <Grid container spacing={3}>
            <Grid item xs={12} lg={12}>
              <JobDescWithImage />
            </Grid>
          </Grid>
        </SuiBox>
        <SuiBox mt={5} mb={3}>
          <Grid container spacing={3}>
            {}

          </Grid>
        </SuiBox>
        <SuiBox mb={1.5}>
          <Grid container spacing={3}>
            <Grid item xs={12} lg={8}>
              <Grid container spacing={3}> 

                {}
              </Grid>
            </Grid>
            {}
          </Grid>
        </SuiBox>
        <SuiBox my={3}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={12}>
              <BillingInformation />
            </Grid>
            {}
          </Grid>
        </SuiBox>
      </SuiBox>
      <Footer />
    </DashboardLayout>
  );
}

export default JobDesc;
