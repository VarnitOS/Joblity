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

// Soft UI Dashboard React components
import SuiBox from "components/SuiBox";
import SuiTypography from "components/SuiTypography";

// Billing page components
import Bill from "layouts/jobdesc/components/Bill";

function BillingInformation() {
  return (
    <Card id="delete-account">
      <SuiBox pt={3} px={2}>
        <SuiTypography variant="h5" fontWeight="medium">
          Reviews
        </SuiTypography>
      </SuiBox>
      <SuiBox pt={1} pb={2} px={2}>
        <SuiBox component="ul" display="flex" flexDirection="column" p={0} m={0}>
          <Bill
            name="oliver liam"
            company="They do a great job at being disability inclusive. I greatly appreciate some of the accessibility services they provide. Things such as dedicated disability support staff give great comfort to employees like myself, knowing that there are staff I can reach out to for any feedback and concerns on disability related issues. It’s been quite enjoyable working with other employees too, most are accommodating, and we operate as equals."
            email="oliverliam@gmail.com"
            vat="FRB1235476"
          />
          <Bill
            name="lucas harper"
            company="It is still common for employers to not provide appropriate employee insurance to people with disabilities, especially a plan that’s comparable to those without disabilities. I’m happy to say that this is not the case here! They are aware of the concerns we have and provide a plan that I am contended with. The company culture is quite inclusive, and while there’s room for improvement on the communication front, they far exceed my past experiences at other companies and deserve recognition."
            email="lucas@gmail.com"
            vat="FRB1235476"
          />
          <Bill
            name="ethan james"
            company="One thing I quite appreciate about working here is both the utility and mental support they provide for employees. Joining the company, I was concerned about needing to pay or provide my own assistive technologies that I need to purchase to perform. However, they have a dedicated team to assist people with disabilities that were helpful and provided the tools I needed. Aside from this, the work environment has been friendly. I think this is a safe option for PWDs to consider."
            email="ethan@gmail.com"
            vat="FRB1235476"
            noGutter
          />
        </SuiBox>
      </SuiBox>
    </Card>
  );
}

export default BillingInformation;
