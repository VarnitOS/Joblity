import { useState, useEffect } from "react";
// @mui material components
import Card from "@mui/material/Card";
import Grid from "@mui/material/Grid";
import Icon from "@mui/material/Icon";

// Soft UI Dashboard React components
import SuiBox from "components/SuiBox";
import SuiTypography from "components/SuiTypography";

// Dashboard layout components
import RecoBox from "layouts/recommender/components/RecoBox";
import Box from "@mui/material/Box";


// Loader Component
import Loader from "react-spinners/ClipLoader";

import deepLearning from "assets/images/deep-learning.png";


function RecoList({ jobs, filterPresent, explanations }) {

  const [jobsDisplayed, setJobsDisplayed] = useState([]);
  const [loading, setLoading] = useState(false);
  const [color, setColor] = useState("navy");

  function shuffle(array) {
    array.sort(() => Math.random() - 0.5);
    return array;
  }

  useEffect(() => {
    if (jobs.length === 0 && !filterPresent) {
      setLoading(true);
    } else if (jobs.length === 0) {
      setJobsDisplayed([])
    }
    else {
      setJobsDisplayed(shuffle(jobs).slice(0, 23));
      setLoading(false);
    }
  }, [jobs]);

  // Setting default values for the props of SuiTypography
  SuiTypography.defaultProps = {
    color: "dark",
    fontWeight: false,
    textTransform: "none",
    verticalAlign: "unset",
    textGradient: false,
    opacity: 1,
  };

  return (
    <SuiBox py={3}>
      {(jobsDisplayed.length === 0 && !filterPresent) ?
        <div style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "50vh" }}>
          <center><h1>Generating Deep Learning Recommendations..</h1>
            <Loader color={color} loading={loading} size={300} margin={0} /></center>
        </div> :
        <>
      {(jobs.length === 0 && filterPresent) && <center><h1>No jobs available.</h1></center>}
          {!filterPresent &&
            <div>
              <div className="center">
                <h1>Deep Learning Recommendations</h1>
              </div>
              <Grid container spacing={3}>
                <Grid item xs={1} sm={1} md={1} lg={1} />
                <Grid item xs={10} sm={10} md={10} lg={10} className="center" mb={4} mt={2}>
                  <Card>
                    <SuiBox p={2} bgColor="#DBF3FA">
                      <Grid container spacing={3}>
                        <Grid item xs={12} lg={12}>
                          <SuiBox height="100%">
                            <SuiBox>
                              <Grid container alignItems="center">
                                <Grid item xs={2}>
                                  <SuiBox ml={2} lineHeight={1}
                                    variant="gradient"
                                    bgColor="info"
                                    color="white"
                                    width="10rem"
                                    height="10rem"
                                    borderRadius="md"
                                    display="flex"
                                    justifyContent="center"
                                    alignItems="center"
                                    sx={{
                                      backgroundImage: ({ functions: { rgba, linearGradient }, palette: { gradients } }) =>
                                        `${linearGradient(
                                          rgba(gradients.light.main, 0.1),
                                          rgba(gradients.light.state, 0.1)
                                        )}, url(${deepLearning})`,
                                      backgroundSize: "contain",
                                      backgroundPosition: "50%",
                                      backgroundRepeat: "no-repeat"
                                    }}>
                                    <></>
                                  </SuiBox>
                                </Grid>
                                <Grid item xs={10}>
                                  <SuiBox ml={1} lineHeight={1}>
                                    <SuiBox pt={1} mb={0.5}>
                                      <SuiTypography fontWeight="medium" color="inherit">
                                        <p>{explanations}</p>
                                      </SuiTypography>
                                    </SuiBox>
                                  </SuiBox>
                                </Grid>
                              </Grid>
                            </SuiBox>
                          </SuiBox>
                        </Grid>
                      </Grid>
                    </SuiBox>
                  </Card>
                </Grid>
              </Grid>
              <Grid container rowSpacing={5} columnSpacing={{ xs: 10, sm: 10, md: 10 }}>
                <Grid item xs={2} sm={2} md={2} lg={2} />
                <Grid item xs={8} sm={8} md={8} lg={8}>
                  {jobsDisplayed.slice(0, 3).map((job) => (
                    <div>
                      <RecoBox
                        key={job.id}
                        company={job.company}
                        title={job.job_title}
                        desc={job.desc.length > 100 ? job.desc.substring(0, 197).concat("...") : job.desc}
                        isTop
                      />
                      <div>&nbsp;</div>
                    </div>
                  ))}
                </Grid>
              </Grid>
            </div>}
            {!filterPresent &&
            <div className="center">
                <h1>Here are some other recommendations for your reference:</h1>
              </div>
            }
          <Grid container spacing={3}>
            {filterPresent && jobsDisplayed.map(job => (
              <Grid item xs={12} sm={6} md={6} lg={4}>
                <RecoBox
                  key={job.id}
                  company={job.company}
                  title={job.job_title}
                  desc={job.desc.length > 100 ? job.desc.substring(0, 97).concat("...") : job.desc}
                />
              </Grid>
            ))}
            
            {!filterPresent && jobsDisplayed.slice(3).map(job => (
              <Grid item xs={12} sm={6} md={6} lg={4}>
                <RecoBox
                  key={job.id}
                  company={job.company}
                  title={job.job_title}
                  desc={job.desc.length > 100 ? job.desc.substring(0, 97).concat("...") : job.desc}
                />
              </Grid>
            ))}
          </Grid>
        </>}
    </SuiBox>
  );
}

export default RecoList;
