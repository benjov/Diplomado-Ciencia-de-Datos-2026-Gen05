# Para conectar a https://login.shinyapps.io/login?redirect=%2F

library(rsconnect)


rsconnect::setAccountInfo(name='XXXX',
                          token='tu-tpken',
                          secret='tu-secret')

deployApp("direccion-en-tu-maquina",
          appName = "NAME-App",
          appTitle = "TITLE-App",
          account = "tu-user")