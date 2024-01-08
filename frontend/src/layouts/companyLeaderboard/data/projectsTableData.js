/* eslint-disable react/prop-types */
// @mui material components
import Icon from "@mui/material/Icon";

// Soft UI Dashboard React components
import SuiBox from "components/SuiBox";
import SuiTypography from "components/SuiTypography";
import SuiProgress from "components/SuiProgress";

// Images
import logoSpotify from "assets/images/small-logos/logo-spotify.svg";
import logoInvesion from "assets/images/small-logos/logo-invision.svg";
import logoSlack from "assets/images/small-logos/logo-slack.svg";
import logoWebDev from "assets/images/small-logos/logo-webdev.svg";
import logoXD from "assets/images/small-logos/logo-xd.svg";
import logoApple from "assets/images/small-logos/logo-apple.svg";
import logoAtlassian from "assets/images/small-logos/logo-atlassian.svg";
import logoFacebook from "assets/images/small-logos/logo-facebook.svg";
import logoGoogle from "assets/images/small-logos/logo-google.svg";
import visa from "assets/images/logos/visa.png";
import mastercard from "assets/images/logos/mastercard.png";

function Completion({ value, color }) {
  return (
    <SuiBox display="flex" alignItems="center">
      <SuiTypography variant="caption" color="text" fontWeight="medium">
        {value}
      </SuiTypography>
      <SuiBox width="8rem">
        <SuiProgress value={value} color={color} variant="gradient" label={false} />
      </SuiBox>
    </SuiBox>
  );
}

export default {
  columns: [
    { name: "rank", align: "center" },
    { name: "company", align: "center" },
    { name: "industries", align: "center" },
    { name: "hired", align: "center" },
    { name: "score", align: "center" },
  ],

  rows: [
    {
      rank: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          1
        </SuiTypography>
      ),
      company: [logoApple, "Apple"],
      industries: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          IT, Software & Services
        </SuiTypography>
      ),
      hired: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          36,786
        </SuiTypography>
      ),
      score: <Completion value={98} color="info" />
    },
    {
      rank: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          2
        </SuiTypography>
      ),
      company: [logoGoogle, "Google"],
      industries: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          IT, Software & Services
        </SuiTypography>
      ),
      hired: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          156,500
        </SuiTypography>
      ),
      score: <Completion value={96} color="success" />
    },
    {
      rank: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          3
        </SuiTypography>
      ),
      company: [logoFacebook, "Facebook"],
      industries: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          IT, Software & Services
        </SuiTypography>
      ),
      hired: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          71,970
        </SuiTypography>
      ),
      score: <Completion value={95} color="error" />
    }, {
      rank: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          4
        </SuiTypography>
      ),
      company: [logoAtlassian, "Atlassian"],
      industries: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          Software
        </SuiTypography>
      ),
      hired: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          6,177
        </SuiTypography>
      ),
      score: <Completion value={92} color="info" />
    },
    {
      rank: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          5
        </SuiTypography>
      ),
      company: [logoSpotify, "Spotify"],
      industries: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          Music Streaming
        </SuiTypography>
      ),
      hired: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          5,587
        </SuiTypography>
      ),
      score: <Completion value={91} color="success" />
    },
    {
      rank: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          6
        </SuiTypography>
      ),
      company: [mastercard, "Mastercard"],
      industries: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          Financial Services
        </SuiTypography>
      ),
      hired: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          18,600
        </SuiTypography>
      ),
      score: <Completion value={90} color="error" />
    },
    {
      rank: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          7
        </SuiTypography>
      ),
      company: [visa, "Visa"],
      industries: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          Financial Services
        </SuiTypography>
      ),
      hired: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          7,500
        </SuiTypography>
      ),
      score: <Completion value={83} color="info" />
    },
    {
      rank: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          8
        </SuiTypography>
      ),
      company: [logoSlack, "Slack"],
      industries: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          Software and Communications
        </SuiTypography>
      ),
      hired: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          1,664
        </SuiTypography>
      ),
      score: <Completion value={81} color="success" />
    },
    {
      rank: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          9
        </SuiTypography>
      ),
      company: [logoWebDev, "Webdev"],
      industries: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          Software
        </SuiTypography>
      ),
      hired: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          875
        </SuiTypography>
      ),
      score: <Completion value={75} color="error" />
    },
    {
      rank: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          10
        </SuiTypography>
      ),
      company: [logoXD, "Adobe"],
      industries: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          Software
        </SuiTypography>
      ),
      hired: (
        <SuiTypography variant="button" color="text" fontWeight="medium">
          24,000
        </SuiTypography>
      ),
      score: <Completion value={68} color="info" />
    }
  ],
};
